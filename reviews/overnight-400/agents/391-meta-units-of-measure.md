# 391 — meta-units-of-measure (SI usage and constants/units alignment)

## Headline
SI base units are used uniformly across all sampled domain packages, but
`constants/units.go` is essentially dead code: zero production references —
every conversion is documented prose, not callable from the library itself.

## Scope of audit

Sampled signatures, godoc, and bodies for ~25 functions across:

- `orbital/orbital.go` — KeplerOrbit, OrbitalPeriod, OrbitalVelocity,
  HohmannTransfer, EscapeVelocity, HillSphere, SynodicPeriod, TrueAnomalyFromMean
- `physics/mechanics.go` — NewtonSecondLaw, ProjectilePosition, GravitationalForce,
  SpringForce, Pendulum, KineticEnergy, PotentialEnergy
- `physics/thermo.go` — IdealGas, StefanBoltzmann, CarnotEfficiency,
  HeatEquation1DStep, FourierHeatConduction, NewtonCooling, ThermalExpansion
- `physics/optics.go` — SnellRefraction, FresnelReflectance, BeerLambertLaw
- `acoustics/acoustics.go` — SoundSpeed, DecibelSPL, SabineRT60, DopplerShift,
  ResonantFrequency, WaveLength, AWeighting
- `em/em.go` — CoulombForce, ElectricField, OhmsLaw, RCTimeConstant,
  ResonantFrequencyLC
- `fluids/fluids.go` — ReynoldsNumber, BernoulliPressure, DarcyWeisbach,
  DragForce, StokesLaw
- `color/spaces.go`, `color/difference.go` — SRGBToLinear, LinearToSRGB,
  LinearRGBToXYZ, XYZToLab, DeltaE2000

## Per-package audit

| Package | Pressure | Frequency | Energy | Mass | Length | Time | Angle | Color domain | Documented? | Issues |
|---|---|---|---|---|---|---|---|---|---|---|
| orbital | n/a | n/a (rad/s implicit via period) | n/a | kg | m | s | rad | n/a | yes (godoc) | semi-major axis a is "m or any consistent unit" in `KeplerOrbit:26` — relaxed for self-similarity, OK |
| physics/mechanics | n/a | n/a | J (KE/PE) | kg | m | s | rad | n/a | yes | clean SI |
| physics/thermo | Pa (IdealGas via R) | n/a | n/a (W power) | n/a | m | s | n/a | n/a | yes | `NewtonCooling:123` and `ThermalExpansion:139` permit "K or C" for *deltas* — physically correct, well-flagged |
| physics/optics | n/a | n/a | n/a | n/a | m | n/a | rad | n/a | yes | clean SI |
| acoustics | Pa | Hz | n/a | kg/mol (M) | m | s | n/a | n/a | yes | clean SI; uses Hz throughout, never kHz |
| em | n/a | Hz (ResonantFrequencyLC) | J (Capacitor/InductorEnergy) | n/a | m | s (RC tau) | n/a | n/a | yes | clean SI; volts/ohms/farads/henries (all SI) |
| fluids | Pa | n/a | n/a | kg | m | s | n/a | n/a | yes | clean SI; rho in kg/m³, mu in Pa·s |
| color | n/a | n/a | n/a | n/a | n/a | n/a | rad internal, deg in CIEDE2000 hue | linear & gamma sRGB both first-class | yes | sRGB transfer functions explicitly piecewise (IEC 61966-2-1); CIEDE2000 reimplements `deg2rad`/`rad2deg` locally (`color/difference.go:130,135`) |

**Verdict on uniformity:** Domain code is fully SI-consistent. Pressure is
always Pa, frequency Hz, energy J, mass kg, length m, time s, angle rad
(degrees only inside CIEDE2000 hue arithmetic where the standard is
defined in degrees). Color cleanly distinguishes linear vs gamma-encoded.

## Critical finding: `constants/units.go` is unused

`grep` for `constants.MetersPer*`, `constants.KgPer*`, `constants.CelsiusToKelvin`,
`constants.FahrenheitTo*`, `constants.RadiansToDegrees`, `constants.DegreesToRadians`,
`constants.SecondsPer*`, `constants.PascalsPer*` across all `*.go` files in
the repo returned **zero matches** outside `constants/` itself
(production hits = 0; only `reviews/` markdown mentions).

The file declares 18 conversion constants (lines 18–101 of
`constants/units.go`):

- `MetersPerMile`, `MetersPerFoot`, `MetersPerInch`, `MetersPerYard`, `MetersPerNauticalMile`
- `KgPerPound`, `KgPerOunce`
- `CelsiusToKelvin`, `FahrenheitToKelvinOffset`, `FahrenheitToKelvinScale`
- `RadiansToDegrees`, `DegreesToRadians`
- `SecondsPerMinute`, `SecondsPerHour`, `SecondsPerDay`
- `PascalsPerAtm`, `PascalsPerBar`, `PascalsPerPSI`

None are referenced by any production package. They exist purely as
constant-value tests in `constants/constants_test.go` (lines 144–274).
Reality functions never need them because every signature is already SI
in/out — by design, conversion is the *caller's* problem.

That's defensible policy, but two consequences:

1. **The constants are easy to drift.** No call site means no compiler
   pressure when one of them is wrong. Today they happen to match
   ISO/BIPM (verified by `constants_test.go`), but a typo would only
   surface if someone ran the constants tests.

2. **`color/difference.go:129–137`** redefines `deg2rad`/`rad2deg` as
   private helpers using `math.Pi/180` literals. This is the only spot
   in the audited surface where a reality function actually needs the
   conversion — and it ignores `constants.DegreesToRadians`. Should use
   `constants.DegreesToRadians` for single-source-of-truth.

## Other observations

- **Duplicate constant.** `constants/physics.go:81` declares
  `AtmPressure = 101325.0` while `constants/units.go:94` declares
  `PascalsPerAtm = 101325.0`. Same value, two names. Pick one.
- **`acoustics/acoustics.go:22`** documents `M: molar mass (kg/mol)` —
  consistent with SI. SoundSpeed uses gas constant R (J/(mol·K)) implicitly
  via `constants.GasConstant`, so the kg/mol unit is required.
- **`physics/thermo.go:123`** intentionally allows "K or C" for absolute
  temperatures in `NewtonCooling` because only the *difference* matters.
  Same pattern in `ThermalExpansion:139`. Both godoc this clearly. Good.
- **`orbital/orbital.go:26,194`** loosen units: `KeplerOrbit`'s `a` is
  "m or any consistent unit"; `SynodicPeriod` accepts "any consistent
  time unit". This is appropriate — these formulas are unit-homogeneous.
- **`color/spaces.go`** — sRGB and linear RGB are both exposed as
  first-class. The transfer functions `SRGBToLinear`/`LinearToSRGB`
  follow IEC 61966-2-1 piecewise (not the simplified γ=2.2 power law),
  which is the right call.
- **No package uses kHz, MHz, kg-force, atm, bar, eV, AU, parsec, day,
  year, or degrees-as-input** in any function signature surveyed.
  Single exception: degrees inside the CIEDE2000 internals (mandated by
  the standard).

## Concrete recommendations

1. **Fix `color/difference.go:129–137`.** Replace the local
   `deg2rad(d)` / `rad2deg(r)` helpers with calls to
   `constants.DegreesToRadians` / `constants.RadiansToDegrees`. This is
   the only reality production site where the units file would naturally
   apply, and it currently ignores it.

2. **Resolve duplicate atmosphere constant.** Either
   - delete `PascalsPerAtm` from `constants/units.go:94` and have callers
     use `AtmPressure` (since "Atm" is the SI quantity), or
   - delete `AtmPressure` from `constants/physics.go:81` and migrate
     callers to `PascalsPerAtm` (matching the unit-conversion family).
   The former matches naming conventions ("AtmPressure" reads like a
   physical reference value, "PascalsPerAtm" reads like a conversion).

3. **Document the policy for `constants/units.go`.** Add a package-level
   comment stating the truth on the ground: "These constants are
   provided for *consumers* of reality. Reality's own functions accept
   and return SI base units exclusively; they never need these
   conversions internally." That makes the zero-call-sites situation a
   feature, not a bug, and discourages future drift toward
   non-SI signatures.

4. **Add cross-units tautology tests.** Already present
   (`RadiansToDegrees * DegreesToRadians == 1`, `5280 * MetersPerFoot
   == MetersPerMile`) — keep these and add `PascalsPerAtm == AtmPressure`
   as a cross-file regression guard until #2 is resolved.

5. **Optional: kelvin/celsius helpers as functions.** The
   Fahrenheit↔Kelvin conversion is a two-step affine transform
   (`(F + 459.67) * 5/9`). Exposing it as raw constants forces every
   consumer to recompose the formula. Adding
   `FahrenheitToKelvin(f) float64` and `KelvinToFahrenheit(k) float64`
   to `constants/units.go` (or a new `constants/convert.go`) would close
   the only ergonomically awkward case. Same applies to mile/foot/etc. if
   those see external use.

## Sources

- `C:/limitless/foundation/reality/constants/units.go` (lines 1–102 — full file)
- `C:/limitless/foundation/reality/constants/physics.go:81` (duplicate AtmPressure)
- `C:/limitless/foundation/reality/constants/constants_test.go:144–274`
  (only call sites for the units constants)
- `C:/limitless/foundation/reality/orbital/orbital.go:26,84,98,153,194,218`
- `C:/limitless/foundation/reality/physics/mechanics.go:39–42,80–83,121–124`
- `C:/limitless/foundation/reality/physics/thermo.go:14–17,30–33,71–75,105–108,121–124,137–140`
- `C:/limitless/foundation/reality/physics/optics.go:15,43`
- `C:/limitless/foundation/reality/acoustics/acoustics.go:18–22,57–59,114,158`
- `C:/limitless/foundation/reality/em/em.go:30–34,71–74,184–193,201–211`
- `C:/limitless/foundation/reality/fluids/fluids.go:19–24,42–46,113–115,164–168,191–193`
- `C:/limitless/foundation/reality/color/spaces.go:16–50,55–84` (sRGB↔linear)
- `C:/limitless/foundation/reality/color/difference.go:129–137` (duplicated deg/rad)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/049-constants-api.md`
  (sibling review on constants surface)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/381-meta-types-system.md`
  (sibling review on type-system options for units)

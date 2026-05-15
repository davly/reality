# 004 — acoustics: API ergonomics

## Headline
The acoustics package's 9 functions are individually well-documented but the surface is internally inconsistent (DecibelSPL vs DecibelFromIntensity, SabineRT60 vs ResonantFrequency, mixed parameter orders) and diverges from sibling packages on naming, error handling, and unit conventions in ways that will surprise a user who has already learned em/ or signal/.

## API surface inventory
| function | params | return | unit conventions |
|---|---|---|---|
| `SoundSpeed` | `gamma, R, T, M float64` | `float64` (m/s) | SI: K, J/(mol·K), kg/mol; constants must be passed in by caller |
| `SoundIntensity` | `P, r float64` | `float64` (W/m²) | SI: W, m |
| `DecibelSPL` | `p, pRef float64` | `float64` (dB) | Pa, Pa; reference must be supplied (no default) |
| `DecibelFromIntensity` | `I, IRef float64` | `float64` (dB) | W/m²; reference supplied |
| `SabineRT60` | `V, A float64` | `float64` (s) | m³, Sabines (m²·α); 0.161 metric constant baked in |
| `DopplerShift` | `f0, vs, vr, c float64` | `float64` (Hz) | Hz, m/s; sign convention in doc only |
| `ResonantFrequency` | `L float64, n int, c float64` | `float64` (Hz) | m, dimensionless, m/s; open-open pipe assumed silently |
| `WaveLength` | `f, c float64` | `float64` (m) | Hz, m/s |
| `AWeighting` | `f float64` | `float64` (dB) | Hz |

9 exported symbols. Zero structs, zero error returns, zero options bags. Every function returns a single `float64` except via `panic` paths (none here).

## Ergonomic problems
- **Naming asymmetry inside the dB pair**: `acoustics.go:65` `DecibelSPL` (suffix qualifier) vs `acoustics.go:81` `DecibelFromIntensity` (preposition phrase). Either both should be `DecibelSPL/DecibelSIL` or both `DecibelFromPressure/DecibelFromIntensity`. Today the user has to remember which idiom each one uses.
- **`ResonantFrequency` silently assumes open-open pipe** (`acoustics.go:149`). A user typing `acoustics.ResonantFrequency` in their IDE has no hint that closed-open (`f_n = (2n-1)c/(4L)`) is unsupported. The em package solves the analogous problem with the explicit suffix `ResonantFrequencyLC` (`em.go:211`); acoustics should mirror this and expose `ResonantFrequencyOpenPipe` (and reserve the bare name or never use it).
- **Parameter ordering is non-uniform with respect to "the medium"**. `WaveLength(f, c)` puts the wave property first, the medium last; `SoundIntensity(P, r)` puts the source first, geometry last; but `ResonantFrequency(L, n, c)` interleaves a geometry parameter, a discrete index, and the medium speed in arbitrary order. There is no rule a user can internalize.
- **`DopplerShift` argument order `(f0, vs, vr, c)`** (`acoustics.go:128`) puts source velocity *before* receiver velocity. Most physics texts (and the formula `(c+vr)/(c+vs)` itself) read receiver-then-source. Worse: `vs > 0` means *receding* but `vr > 0` means *approaching*. Two parameters of identical type and units with opposite sign conventions, undistinguished in the type signature, is a foot-gun.
- **`SabineRT60` accepts `A` in "Sabine units" without a type alias** (`acoustics.go:101`). `m²` is not actually `m²` here — it's `Σ αᵢSᵢ` (absorption coefficient × surface area summed). A user who confuses room surface area with Sabine absorption gets a wrong answer with no signal.
- **No constants for reference values**. `pRef = 20e-6 Pa` and `IRef = 1e-12 W/m²` are mentioned in docstrings but not exported. Compare to em where `coulombConst` is private but the user never has to type it. The acoustics user will type `20e-6` everywhere or define their own.
- **Inputs that the docstring says are invalid are not validated**. `SoundSpeed(gamma=-1, ...)` returns NaN per `math.Sqrt`, but `SabineRT60(V=-5, A=10)` returns a negative time silently; `ResonantFrequency(L=-1, n=1, c=343)` returns a negative frequency silently; `DecibelSPL(p=-1, pRef=2e-5)` returns NaN — three different failure modes for "negative input you said wasn't allowed". (See agent 001 finding for the silent-negative-input issue in detail.)
- **`AWeighting` returns dB but takes Hz, while every other function returning dB (`DecibelSPL`, `DecibelFromIntensity`) takes a *ratio*.** User has to mentally categorize: "is this the curve, or the level?" — would be clarified by `AWeightingDB(f)` or grouping in a `weighting` subpackage.
- **No frequency-domain I/O type**. Adjacent functions in signal/ accept `[]float64` buffers; acoustics has nothing that operates on a spectrum. A user who wants to apply A-weighting to an FFT bin array writes their own loop.

## Naming consistency
- 6 of 9 names are noun phrases (`SoundSpeed`, `SoundIntensity`, `WaveLength`, `ResonantFrequency`, `AWeighting`, `SabineRT60`). 2 are verb-suffixed nouns (`DopplerShift`, `DecibelSPL` if you read SPL as the noun). 1 is a prepositional construction (`DecibelFromIntensity`). No consistent rule.
- `SabineRT60` is the only function name that embeds a person + a quantity acronym. The em package uses `OhmsLaw` (`em.go:78`) — person + the word "Law" — and `RCTimeConstant` (`em.go:194`) — components + quantity. Acoustics should pick one.
- `WaveLength` is unconventional Go capitalization; the standard English is "wavelength" and the Go name should be `Wavelength` (single word). Compare `AWeighting` (which is correctly one logical token).
- "Decibel" prefix is reserved exclusively for the two dB-scale converters but not for `AWeighting`, even though `AWeighting` returns dB. A user grepping for `Decibel` will miss it.

## Cross-package comparison
- **vs `signal/`**: signal/ uses caller-provided output buffers (`Convolve(signal, kernel, out []float64)`, `HannWindow(n int, out []float64)`) and `panic` on bad inputs. acoustics/ returns a scalar and uses NaN/Inf for bad inputs. The styles are reasonable for their respective use cases (scalar vs vector), but signal/ also lists `Consumers:` in every doc comment (`fft.go:47`, `filter.go:53`) — acoustics never does, so a reader cannot tell which functions Pistachio actually calls at 60 FPS.
- **vs `physics/`**: physics/ uses tuple returns where natural (`ProjectilePosition` returns `(x, y)`; `ElasticCollision` returns `(v1f, v2f)` — `mechanics.go:47, 103`). acoustics never uses multi-return even where it would help (e.g., `DopplerShift` could return `(f, wavelengthShift)` or a `DopplerResult` struct disambiguating sign convention). physics/ also uses bare constants from `constants/` (e.g., `constants.GravitationalConst`); acoustics imports nothing from constants and forces the caller to supply `R`, `gamma`, `M`, `c` every call, which is verbose for the dominant air-at-STP use case.
- **vs `em/`**: em/ has `ResonantFrequencyLC` (`em.go:211`) — explicit suffix names the topology. acoustics has bare `ResonantFrequency` — implicit topology. em/ also derives Coulomb's constant once at package init (`em.go:21`) so the caller never types ε₀; acoustics could analogously expose `SoundSpeedAir(T)` deriving γ, R, M from constants.
- **vs all three**: every sibling has either tuple returns, slice I/O, or constants from `constants/`. acoustics has none of these. It looks like the first package written before the conventions stabilized.

## Concrete recommendations
1. **Rename for symmetry**: `DecibelFromIntensity` → `DecibelSIL` (matching `DecibelSPL`), or rename both to `DecibelFromPressure` / `DecibelFromIntensity`. Pick one idiom.
2. **Topology in the name**: `ResonantFrequency` → `ResonantFrequencyOpenPipe`; reserve room for `ResonantFrequencyClosedPipe`, `ResonantFrequencyHelmholtz`, `ResonantFrequencyMembrane`. Mirrors `em.ResonantFrequencyLC`.
3. **`Wavelength` not `WaveLength`** — single English word, fixes search/grep ergonomics.
4. **Provide air-at-STP convenience**: `SoundSpeedAir(T float64) float64` calling `SoundSpeed(1.4, constants.GasConstant, T, 0.02896)`. The vast majority of acoustics callers want air, not arbitrary gas.
5. **Export the canonical references**: `const (ReferencePressureAir = 20e-6; ReferenceIntensityAir = 1e-12)` so users don't retype magic numbers. Add `DecibelSPLAir(p)` and `DecibelSILAir(I)` zero-arg-ref convenience wrappers.
6. **Pick one bad-input contract** package-wide. Sibling packages already split: signal/ panics, physics/em/ return NaN/Inf. Acoustics should return NaN for negative inputs that the docstring says are invalid (currently it silently returns negative frequencies/times). At minimum, `if L <= 0 { return math.NaN() }` in `ResonantFrequency`, `WaveLength`, `SabineRT60`.
7. **Add a unit-bearing type alias for Sabine absorption**: `type Sabines = float64` (or a wrapper struct) so `SabineRT60(V, A Sabines)` discourages passing raw surface area.
8. **Reorder `DopplerShift` to `(f0, vSource, vReceiver, c)` and unify sign convention** — both velocities positive when the object is moving toward the other party. Today's mixed convention is the kind of thing that ships shipped wrong-pitch sirens.
9. **List `Consumers:` in each doc comment** as signal/ does (`fft.go:47`). Tells the reader whether the function is hot-path or analytic.
10. **Group A-weighting alongside future B/C/D/Z curves** in either an `acoustics/weighting` subpackage or a clear `Weighting` prefix (`WeightingA(f)`, `WeightingC(f)`); current name pattern `AWeighting` doesn't scale (`BWeighting`, `CWeighting` collide alphabetically with unrelated things).

## Sources
- C:/limitless/foundation/reality/acoustics/acoustics.go (entire 197 lines)
- C:/limitless/foundation/reality/em/em.go:1-214 (sibling comparison: naming suffix, derived constants, topology in name)
- C:/limitless/foundation/reality/physics/mechanics.go:47, 103 (tuple-return convention)
- C:/limitless/foundation/reality/physics/optics.go:21-26 (NaN-on-invalid convention)
- C:/limitless/foundation/reality/signal/fft.go:37-67 (panic-on-invalid + Consumers: tag convention)
- C:/limitless/foundation/reality/signal/filter.go:19-40 (output-buffer convention)
- C:/limitless/foundation/reality/signal/window.go:1-32 (n + out buffer parameter ordering)

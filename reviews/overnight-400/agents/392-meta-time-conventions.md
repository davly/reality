# 392 — meta-time-conventions (TT/TAI/UTC/TDB across reality)

## Headline
Reality has no time-scale concept anywhere in production code: every package treats time as a dimensionless `float64` (relative dt, sample index, anomaly radians); the recommended policy is **stay relative-time-only as the library default** and add a small, isolated `orbital/timescales.go` (TT−TAI = 32.184 exact, TDB↔TT linear, IAU 2006 Res B3 constants, no leap-second table) only when an actual orbital propagator or ephemeris consumer lands — never in `signal`, `audio`, `physics`, or `chaos`.

## Per-package time semantics (audit, file:line)

| Package | How time enters | Type | Implicit scale | Notes |
|---|---|---|---|---|
| `orbital/orbital.go` | mean/true anomaly (rad), period T, dt-free | `float64` | none — frame-blind, scale-blind | `KeplerOrbit` takes `nu` not `t`; `OrbitalPeriod` returns *seconds* of whatever scale `mu` was integrated in (TDB if μ from DE441, TT if from IAU 2015 nominal). Never tagged. orbital.go:84,201 |
| `physics/mechanics.go` | `t` in `ProjectilePosition(v0,θ,t,g)`, dt-implicit elsewhere | `float64` | Newtonian (absolute) | Pure classical mechanics — no special relativity, no proper time, no Lorentz. mechanics.go:47 |
| `physics/thermo.go`, `physics/optics.go`, `physics/materials.go` | none | — | — | Time-free (steady-state thermo, ray optics, material props) |
| `chaos/systems.go` | `t float64` is ODE independent variable | `float64` | abstract | Lorenz/Rössler/SIR/VdP all take `t` but their RHS is autonomous → `t` could be removed. Standard Go ODE signature retained for non-autonomous future use. systems.go:17,40,67,93,115 |
| `calculus/*` | `t` in integrand, `dt` step | `float64` | abstract | Numerical quadrature/RK4 — pure mathematical time |
| `signal/fft.go` | `sampleRate` (Hz) + sample index | `float64` Hz, `int` index | relative t=0 | `FFTFrequencies(n, sampleRate, out)` computes `freq[k]=k·sr/N`. No absolute time. fft.go:163,167,176 |
| `signal/filter.go`, `signal/window.go` | sample index only | `int` | relative | No timestamps |
| `acoustics/acoustics.go` | RT60 in *seconds*, frequency in Hz | `float64` | relative | Sabine reverb, Doppler, dB SPL — physical durations, no calendar |
| `audio/melscale.go`, `audio/cqt`, `audio/pitch/*` | `sampleRate`, `frameSize`, `hopSize` | `float64`, `int` | sample-relative | Frame index `t` × hopSize gives sample offset; never wall-clock. vad_based.go:23-28 |
| `audio/segmentation/*` | `Segment{StartIdx, EndIdx}` in samples | `int` | sample-relative | Caller divides by sampleRate to get seconds. Zero leak of `time.Time`. |
| `control/*` | dt, settling time, frequency response | `float64` | relative | Pure transfer-function math |
| `prob/*`, `queue/*`, `gametheory/*`, `linalg/*`, `crypto/*`, `combinatorics/*`, `compression/*`, `color/*`, `geometry/*`, `graph/*`, `optim/*`, `em/*`, `fluids/*` | none or abstract dt | — | — | No notion of absolute time |
| `constants/units.go` | `SecondsPerMinute=60`, `SecondsPerHour=3600`, `SecondsPerDay=86400.0` | const float64 | **untagged** | Comment says "mean solar day" — accurate for UT1, off by ≤1 s/day for UTC, exact for TAI/TT by definition. units.go:81-88. Slot 046 already filed C-TIME-1 (annotate scale) and C-TIME-2 (add `SecondsPerJulianYear=31557600.0`). |
| `constants/physics.go` | c, h, e, k, N_A | exact | — | No `Δν_Cs = 9192631770 Hz` (slot 378 gap), no `JulianDateJ2000 = 2451545.0`, no IAU 2015 nominal constants (slot 368 gap). |
| `conduit/emit.go:60` | `time.Now().UTC().Format(time.RFC3339)` | `time.Time` | UTC | **Only** `time.Time` use in any non-test `.go` file — and it's an event-emitter helper, not math. Acceptable. |

## Summary of facts

1. **Zero `time.Time` in any math package.** Production code uses `time.Time` only at `conduit/emit.go:60` for log timestamps. No leap-second tables. No JD↔Gregorian. No timezone code.
2. **All "time" inputs are dimensionless `float64`s** carrying whatever scale the caller chose. Compiler cannot catch a UTC value passed to a TT-expecting function — a bug class that *cannot exist today* because no function expects any specific scale.
3. **No relativistic time anywhere.** `physics` is purely Newtonian; no proper time, no `gamma factor`, no Lorentz transform (separately observed in slot search).
4. **Slot 368 confirmed:** `orbital` is frame-blind and scale-blind. Slot 107 (orbital-missing) drafted T1.9 (`TAIfromUTC`/`TTfromTAI`/`GPSfromUTC`, ~150 LOC + 37-row leap-second table) and T3.8 (TDB Fairhead-Bretagnon, ~400 LOC, 1.66-ms annual term).
5. **Slot 378 confirmed:** CGPM-27 Res 4 retires leap seconds by 2035 → TAI−UTC freezes at +37 s (or possibly +36 if a negative leap lands first). Reality's `SecondsPerDay = 86400` becomes *exactly* the TT/TAI/UT1/UTC second by 2035 — i.e. the current implicit scale becomes correct-by-default. **The looming SI change actually validates the relative-time-only stance.**
6. **Slot 109 already drafted the type-tagged design:** `type TimeScale uint8 { UNSPEC, TT, TAI, UTC, UT1, TDB, GPS }` — ready to lift if/when needed.

## Recommendation: time-scale policy for reality

### Tier 0 — adopt now (zero LOC, doc-only, v0.11)

**Reality is a relative-time math library.** Document this explicitly in `CLAUDE.md` and in each top-of-package doc comment that uses time:

> Time inputs are abstract `float64` durations or sample indices in caller-chosen units. Reality does not bind to TT, TAI, UTC, UT1, TDB, TCB, TCG, or GPS time scales. Callers needing scale conversions (leap seconds, TT−TDB periodic offset) must layer those on top — see `aicore/time/` or equivalent.

This is a **policy** statement, costs nothing, and prevents any future leak of `time.Time` into math packages.

### Tier 1 — small constants (≤20 LOC, v0.11)

Add to `constants/units.go` with explicit scale tags in comments (matches slot 046 C-TIME-1, C-TIME-2; slot 047 list):

```go
// SecondsPerDay is one day in TAI/TT/UT1 (mean-solar) seconds.
// In UTC, a day may contain 86399, 86400, or 86401 SI seconds when a
// leap second is inserted; CGPM-27 Resolution 4 (2022) phases out leap
// seconds by 2035, after which UTC days are also exactly 86400 s.
const SecondsPerDay = 86400.0

const SecondsPerJulianYear   = 31557600.0    // 365.25 × 86400, exact, IAU light-year
const SecondsPerJulianCentury = 3155760000.0 // 36525 × 86400, exact
const JulianDateJ2000 = 2451545.0            // 2000-01-01 12:00:00 TT, IAU 2006
```

No conversion functions. No leap-second table. Pure named constants.

### Tier 2 — IAU time-scale offsets as constants (≤30 LOC, v0.11–v0.12)

Add to `constants/physics.go` (or new `constants/iau.go`), no logic — just IAU 2006 Resolution B3 defining numbers (slot 368, slot 378):

```go
// TTminusTAI is TT − TAI = 32.184 s exactly (IAU 2006 Res B3 §1).
const TTminusTAI = 32.184

// LB is the TDB−TCB rate constant (IAU 2006 Res B3 §3).
const LB = 1.550519768e-8

// LG is the TT−TCG rate constant (IAU 2006 Res B3 §2).
const LG = 6.969290134e-10

// TDB0 is the TDB epoch offset (IAU 2006 Res B3).
const TDB0 = -6.55e-5

// T0 is the TDB/TCB defining epoch JD (TT) = 2443144.5003725.
const T0 = 2443144.5003725

// LeapSecondsTAIminusUTC2026 is TAI − UTC = +37 s (since 2017-01-01,
// no further leap seconds through 2026; possibly negative leap before 2035;
// frozen after CGPM-27 Res 4 takes effect ≤ 2035).
const LeapSecondsTAIminusUTC2026 = 37
```

These are *numbers* (IAU first-principles definitions), not data tables — they fit reality's "encode universal truth" charter without violating zero-data rule.

### Tier 3 — `orbital/timescales.go` (only if propagator lands, v0.12+, ~120 LOC)

**Defer until a real propagator (slot 107 T1.1 universal-variable) or ephemeris reader (slot 368 Chebyshev evaluator) is added.** Then ship:

- `TTfromTAI(jdTAI float64) float64` — adds `TTminusTAI` constant.
- `TAIfromTT(jdTT float64) float64` — subtracts.
- `TDBfromTT(jdTT float64) float64` — Fairhead-Bretagnon 10-term truncation, ~2 ms peak accuracy (slot 107 T3.8 full 1000-term version is overkill for v1; ship truncated). Linear IAU 2006 Res B3 part is exact.
- `TCBfromTDB(jdTDB float64) float64` — exact linear via LB, T0, TDB0.
- **No leap-second table.** UTC↔TAI requires the IERS Bulletin C history (37 entries 1972–2017), which is *data, not math.* That belongs in aicore or a sibling `astro` package — same logic as slot 368's "no SPK in reality."
- **Optional type tags** (slot 109's design): `type JulianDate float64; type TT JulianDate; type TAI JulianDate;` — compiler-checked. Lift only when ≥3 functions consume tagged times; one function alone doesn't justify the API surface.

### Never (out of charter)

- **No `time.Time` in any math package.** It carries timezone and monotonic-clock state that breaks pure-function determinism. Reuse `conduit`'s pattern: `time.Time` only at I/O boundaries.
- **No leap-second tables in reality.** They're observational data (IERS C04 / Bulletin C), not first-principles math. Violates rule 6 (slot 368).
- **No relativistic time-scale conversions in `physics/`.** TT/TCG/TCB rates are *barycentric/geocentric coordinate-time* concepts that belong with frames in `orbital/`, not with classical mechanics.
- **No Coordinated Lunar Time** (slot 378 §6) until CGPM-28 ratifies a definition (Oct 2026). Currently a +56 µs/day offset relative to TT — defer to v1 + 12 mo at minimum.

## Cross-references

- Slot 046 (constants-numerics) — C-TIME-1, C-TIME-2 patches drafted.
- Slot 047 (constants-missing) — `SecondsPerJulianYear`, `JulianDateJ2000`, etc. listed.
- Slot 048 (constants-sota) — S-META-7 leap-second caveat patch drafted.
- Slot 106 (orbital-numerics) — flagged time conversions as "Absent" null finding.
- Slot 107 (orbital-missing) — T1.9 (~150 LOC) and T3.8 (~400 LOC) drafts.
- Slot 108 (orbital-sota) — argued *against* Orekit's god-class `AbsoluteDate`; agree.
- Slot 109 (orbital-api) — `TimeScale` enum + tagged-type design ready to lift.
- Slot 368 (research-iau-frames) — proposed `orbital/timescales.go` with IAU 2006 Res B3 constants.
- Slot 378 (research-survey-physical) — leap-second phase-out 2035, LTC roadmap, CGPM-28 agenda.

## Verdict

The current "all time is `float64` seconds, scale unspecified" stance is **correct and should be preserved as the library default**. Reality is a math library, not a timekeeping library. Add the IAU 2006 Res B3 named constants (Tier 1+2, ~30 LOC, zero behaviour) so consumers can build correct conversion layers above reality without re-deriving 32.184 or 1.550519768e-8 themselves. Defer `orbital/timescales.go` (Tier 3) until an actual orbital propagator demands it — and even then, do not ship a leap-second table.

## Sources

- `C:\limitless\foundation\reality\orbital\orbital.go` (frame-blind, scale-blind Kepler/Hohmann/Vis-viva)
- `C:\limitless\foundation\reality\physics\mechanics.go:47` (`ProjectilePosition` takes `t` Newtonian)
- `C:\limitless\foundation\reality\chaos\systems.go:17,40,67,93,115` (ODE `t` abstract)
- `C:\limitless\foundation\reality\signal\fft.go:163-176` (`sampleRate` only; relative t=0)
- `C:\limitless\foundation\reality\audio\segmentation\vad_based.go:23-28` (sample-index segments)
- `C:\limitless\foundation\reality\constants\units.go:81-88` (`SecondsPerMinute/Hour/Day`, untagged)
- `C:\limitless\foundation\reality\conduit\emit.go:60` (only `time.Now().UTC()` in repo, log helper)
- `reviews/overnight-400/agents/046-constants-numerics.md` (C-TIME-1, C-TIME-2)
- `reviews/overnight-400/agents/107-orbital-missing.md` (T1.9, T3.8 drafts)
- `reviews/overnight-400/agents/109-orbital-api.md` (`TimeScale` enum)
- `reviews/overnight-400/agents/368-research-iau-frames.md` (IAU 2006 Res B3, DE441 stance)
- `reviews/overnight-400/agents/378-research-survey-physical.md` (CGPM-27 R4 leap-second phase-out, LTC)

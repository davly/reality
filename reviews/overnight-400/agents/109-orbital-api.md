# 109 — orbital: API Ergonomics (state representation, frames, time, body)

**Scope.** API shape of `C:/limitless/foundation/reality/orbital/orbital.go` (267 LOC, 8 exported functions). Topic, per MASTER_PLAN: *state vector vs orbital element, frame transforms (ICRF, ITRF)*. Companion to 106 (numerics — done), 107 (missing features — done), 108 (SOTA library porting — done). This audit deliberately avoids those three angles and asks one question: **given the existing 8 functions and any future expansion, what should the *type signatures* look like?** The answer touches state representation, frame tagging, time-scale tagging, body/μ provenance, and consistency with sibling packages (`physics`, `geometry`, `chaos`, `linalg`).

**Headline finding.** The package is **untagged-float64-everywhere** and that is internally consistent with the rest of `reality` — but it is also the principal source of the silent foot-guns flagged in 106 (e≥1 fed to elliptic Newton, ν fed to KeplerOrbit when r is hyperbolic-divergent) and the principal blocker for 107's Tier-1 backlog (no place to hang a `(r, v) ↔ (a,e,i,Ω,ω,ν)` converter, because there is no canonical 6-tuple type). The single most leverageable API decision in this package is **what to call a state vector**: stay numbers-in-numbers-out (status quo), introduce a `[6]float64` ordered tuple per representation (cheapest), or introduce one `OrbitState` struct with explicit fields and frame/time tags (most ergonomic). I argue below for the **middle path** with a clear migration runway.

---

## 1. Status quo — 8 free functions, all `float64`

| Function | Signature | State representation |
|---|---|---|
| `KeplerOrbit` | `(a, e, i, ω, Ω, ν float64) → (x, y, z float64)` | COE → position only (no velocity) |
| `OrbitalPeriod` | `(a, mu) → float64` | scalar |
| `OrbitalVelocity` | `(mu, r, a) → float64` | scalar (vis-viva, no direction) |
| `HohmannTransfer` | `(r1, r2, mu) → (dv1, dv2)` | scalar |
| `EscapeVelocity` | `(M, r) → float64` | scalar; uses `constants.GravitationalConst` internally |
| `HillSphere` | `(a, m, M) → float64` | scalar |
| `SynodicPeriod` | `(T1, T2) → float64` | scalar |
| `TrueAnomalyFromMean` | `(M, e, maxIter int) → ν float64` | scalar |

**What is *not* in any signature today.** No `Frame` enum, no `TimeScale` enum, no `Body` parameter struct, no `StateVector` struct, no `OrbitalElements` struct, no `[3]float64` position vector (KeplerOrbit returns three named scalars `x, y, z`), no `[6]float64` state. The only struct in any orbital file is implicit — none. Compare `chaos/ode.go` which exposes `[]float64` state vectors, or `geometry/quaternion.go` which uses `[3]float64`/`[4]float64` arrays consistently.

**Mu provenance.** Six of eight functions take `mu` as a free parameter; `EscapeVelocity` is the lone exception, hard-wiring `constants.GravitationalConst` and asking for `M` (kg) instead. This is **inconsistent** — see §6 below.

---

## 2. The COE ↔ RV gap: `KeplerOrbit` is half a converter

`KeplerOrbit(a,e,i,ω,Ω,ν) → (x,y,z)` performs the **forward** direction of the classical-orbital-elements-to-radius-vector transform. The package is silent on:

1. **Velocity output.** The same 3-1-3 rotation that produces `r̄_inertial` also produces `v̄_inertial` from `(ḟ, ṙ)` in the perifocal frame; refusing to return velocity costs nothing extra (the trig is already cached) and forces every downstream caller to reimplement perifocal → inertial. Vallado §2.5 Algorithm 10 is one routine returning **both** `r` and `v`. **Recommendation A1:** rename `KeplerOrbit` → `COEtoRV` and return `(r, v [3]float64)` (or `(rx,ry,rz, vx,vy,vz float64)` to stay scalar-tuple).

2. **Reverse transform `RV → COE`.** Vallado §2.5 Algorithm 9 — a 60-line routine using cross products of `r̄, v̄, h̄ = r̄×v̄, n̄ = ẑ×h̄`. Without it, every Tier-1 propagator (107-T1.1 universal-variable Kepler, 107-T1.5 Lambert) has to publish its results as `(r,v)` and then the user *cannot* round-trip to elements. **Recommendation A2:** add `RVtoCOE(r, v [3]float64, mu float64) (a, e, i, capOmega, omega, nu float64)`.

The existence of a forward-only converter is one of the cleanest "API smell" cases in the entire `reality` repo: the function name `KeplerOrbit` even buries the fact that it is a coordinate transform — it sounds like a propagator. **Recommendation A3:** keep `KeplerOrbit` as a deprecated alias that calls `COEtoRV` and discards velocity; document the rename.

---

## 3. State representation — three options, ranked

### Option 0 — keep scalar arguments (status quo)
- **Pros.** Aligns with the rest of `reality` ("numbers in, numbers out"). No allocation. Cross-language golden files trivial.
- **Cons.** Argument order memorisation (`KeplerOrbit(a, e, i, omega, capOmega, nu)` — *is* it `ω` then `Ω` or `Ω` then `ω`? The function takes them in a non-standard order!). Six positional `float64` is a documented API anti-pattern (Vallado §2.5 reorders them). No place to hang frame/time/body metadata.

### Option 1 — `[6]float64` arrays per representation
```go
type COE = [6]float64    // {a, e, i, Ω, ω, ν}
type RV  = [6]float64    // {rx, ry, rz, vx, vy, vz}

func COEtoRV(coe COE, mu float64) RV { ... }
func RVtoCOE(rv RV, mu float64)  COE { ... }
```
- **Pros.** Zero allocation, value semantics, matches `geometry`'s `[3]float64`/`[4]float64`/`linalg.Vec3` conventions. Indexable for cross-language tests (Python `coe[0]` ↔ Go `coe[0]`).
- **Cons.** No compile-time distinction between the two representations (both are `[6]float64`); the type alias `=` does not create a new type, so the compiler will let you pass an RV where a COE is expected. Without a defined type (no alias), this is fragile.
- **Refinement (1b).** Use named types: `type COE [6]float64` and `type RV [6]float64`. Now the type system blocks the foot-gun, at the cost of `coe[i]` indexing remaining ergonomic. **This is the cheapest type-safe option.**

### Option 2 — explicit struct with optional frame/time tags
```go
type StateVector struct {
    R, V  [3]float64    // position, velocity
    Mu    float64       // gravitational parameter of central body
    Frame Frame         // ICRF | ITRF | GCRF | EME2000 | TEME | unspecified=0
    Epoch float64       // seconds since J2000.0 in TimeScale below
    Scale TimeScale     // TT | TAI | UTC | TDB | UT1 | unspecified=0
}

type OrbitalElements struct {
    A, E, I, RAAN, ArgPeri, TrueAnomaly float64
    Mu    float64
    Frame Frame
    Epoch float64
    Scale TimeScale
}
```
- **Pros.** Self-documenting, IDE-completion-friendly, future-proof for SGP4/perturbation propagators (107 backlog Tier-1/2). Frame mismatch becomes a runtime check (`if a.Frame != b.Frame { panic }`) instead of a silent error.
- **Cons.** Departs from `reality`'s "scalars in, scalars out" convention — only `chaos.BifurcationPoint`, `control.PIDController`, `control.TransferFunction`, and a few others use structs. Allocation cost is zero (value type), but pointer-vs-value receiver convention has to be set. Cross-language golden files become harder (Python has no `StateVector`; we'd serialise as JSON object with named keys instead of positional array).

### Recommendation — **Option 1b** for now, with **Option 2** as the v0.11+ migration target

The Tier-1 backlog (107-T1.1 universal-variable Kepler, 107-T1.5 Lambert solver, 107-T1.6 perifocal frame, 107-T1.7 RAAN drift) all consume and produce `(r̄, v̄)`. Shipping them on bare `[3]float64` r and `[3]float64` v is fine; promoting to `RV [6]float64` named type closes the COE/RV mismatch foot-gun with zero allocation. The struct upgrade is a v0.11.0 break and should be done **once**, after the universal-variable propagator and Lambert lock the canonical state representation.

---

## 4. Frame tagging — the ICRF/ITRF question

**Today.** No frame information is attached to any returned coordinate. `KeplerOrbit` returns `(x, y, z)` in *some* inertial frame — the doc says "inertial frame" with no further specification. The 3-1-3 Euler rotation `(Ω, i, ω)` is from the *orbit's* perifocal frame to *its* parent inertial frame — the function is body-agnostic, so calling it with Earth-orbit elements gives ECI coordinates (specifically the frame in which `i` and `Ω` are referenced — typically EME2000 / J2000 / GCRF), and calling it with heliocentric elements gives heliocentric ecliptic coordinates. The function **cannot tell which** because the input parameters do not encode the frame.

**The minimum viable frame system** (to be added when the first frame-transform function ships, see 107-T2.x):
```go
type Frame uint8
const (
    FrameUnspecified Frame = 0
    ICRF             Frame = 1  // International Celestial Reference Frame (≈ J2000 inertial)
    GCRF             Frame = 2  // Geocentric Celestial Reference Frame (Earth-centred ICRF)
    EME2000          Frame = 3  // Mean Equator/Equinox of J2000 (≈ GCRF, ~mas offset)
    ITRF             Frame = 4  // International Terrestrial Reference Frame (Earth-fixed)
    TEME             Frame = 5  // True Equator Mean Equinox (SGP4 native frame)
    Perifocal        Frame = 6  // PQW frame of an orbit (in-plane)
    Heliocentric     Frame = 7  // Solar-system-barycentric ecliptic
)
```
**Why a `uint8` enum, not a string.** Cross-language golden files: a JSON `"frame": 3` is bit-identical Go/Python/C++/C#, while `"frame": "EME2000"` requires every consumer to maintain a name table. This matches `reality`'s existing convention (no string IDs anywhere in the public API).

**Frame-transform function shape.**
```go
func TransformFrame(r [3]float64, from, to Frame, jdTT float64) [3]float64
func TransformFrameFull(r, v [3]float64, from, to Frame, jdTT float64) (rOut, vOut [3]float64)
```
The Earth-orientation parameters needed for ICRF↔ITRF (precession, nutation, polar motion, UT1-UTC) are time-dependent — hence `jdTT` argument. The IAU 2006 precession + IAU 2000A nutation series (~1300 terms, MIT-licensed reference in IERS Conventions) is the canonical implementation; Vallado Ch. 3 gives a closed-form approximation good to 1 m at LEO. **Without** a time argument any "frame transform" is wrong — and the absence of a time argument in the existing API is the cleanest signal that frames have not been thought about yet.

**Recommendation B.** Defer frame tagging until 107-T2 (frame transforms) lands; when it does, the `Frame` enum + `TransformFrame*` functions should be in `orbital/frames.go` and **every** function that returns position/velocity should be updated to either (a) document its output frame in the doc comment as a string identifier, or (b) optionally take a `Frame` parameter for the inertial frame the COE are referenced in. Don't half-do this — either every position-returning function tags its frame or none do.

---

## 5. Time-scale tagging — TT vs UTC vs TDB

**Today.** `OrbitalPeriod`, `SynodicPeriod` return seconds (or any consistent unit). `TrueAnomalyFromMean` consumes mean anomaly (radians) — no time. The package has **no time argument anywhere**, which is correct for the eight closed-form functions (none of them depend on absolute epoch).

**The moment a propagator (107-T1.1) is added, this changes.** A universal-variable Kepler propagator takes `Δt` (a duration, no scale needed); but SGP4 (107-T2.x) consumes a TLE epoch which is **UTC**, propagates internally in **TT/TAI**, and outputs in **TEME** at a **UTC** time the user usually wants converted to **UT1** for ground-station pointing. The five-time-scale dance (TT, TAI, UTC, UT1, TDB) is *the* defining complexity of operational astrodynamics.

**Recommendation C.** When time enters the API, follow Astropy/Orekit precedent:
```go
type TimeScale uint8
const (
    TimeUnspecified TimeScale = 0
    TT              TimeScale = 1  // Terrestrial Time (continuous, theoretical)
    TAI             TimeScale = 2  // International Atomic Time (TT - 32.184 s)
    UTC             TimeScale = 3  // Coordinated Universal Time (leap-seconded)
    UT1             TimeScale = 4  // Mean solar time (Earth rotation)
    TDB             TimeScale = 5  // Barycentric Dynamical Time (≈ TT, periodic offset)
    GPS             TimeScale = 6  // GPS time (TAI - 19 s, no leap seconds)
)

// All time arguments in orbital/* are float64 Julian Dates with a TimeScale tag.
// When the tag is TimeUnspecified, functions either pass through (durations) or
// document a default (typically TT for theory, UTC for I/O).
```
A leap-second table is unavoidable for UTC↔TAI and is **dependency-creep risk** — IERS bulletin C is updated every six months. **Mitigation:** ship the leap-second table as a Go-source constant slice (last value as of release date) plus a `RegisterLeapSecond(jdMjd int, tai_minus_utc int)` hook so applications can update without recompiling `reality`. This is what `chrono` does in C++ and what Astropy does via IERS update lookups; the static-table-with-runtime-extension pattern keeps zero-dependency at compile time.

**Don't ship time-scale tagging until propagator ships.** Premature.

---

## 6. Mu / body parameter — fix the `EscapeVelocity` outlier

**Today's inconsistency.**

| Function | Body input | Note |
|---|---|---|
| `OrbitalPeriod(a, mu)` | μ directly | ✓ |
| `OrbitalVelocity(mu, r, a)` | μ directly | ✓ but argument order differs from above (`mu` first, not last) |
| `HohmannTransfer(r1, r2, mu)` | μ directly | ✓ |
| **`EscapeVelocity(M, r)`** | **kg directly + uses `constants.GravitationalConst`** | **inconsistent** |
| `HillSphere(a, m, M)` | two kg masses | uses ratio so G cancels — fine |

`EscapeVelocity` is the only function that consumes `M` (kg) instead of `μ` (m³/s²). Per the package doc:

> The gravitational parameter μ = G·M is used throughout rather than separate G and M, as μ is typically known to higher precision for celestial bodies.

— `EscapeVelocity` violates its own package contract. The fix is one line:

**Recommendation D1.** Rename or break to `EscapeVelocity(mu, r float64) float64 { return math.Sqrt(2.0 * mu / r) }`. The relative uncertainty in μ_⊕ is ~10⁻¹², in G it is 2.2×10⁻⁵, so the kg-and-G version is ~7 decimal digits worse for *any* real body. Migration: keep `EscapeVelocityFromMass(M, r)` for the kg case during a deprecation cycle.

**Recommendation D2.** Standardise argument order. Two reasonable conventions exist in textbooks:

- **State-first**: `(r, mu)`, `(a, mu)`, `(r1, r2, mu)` — current `OrbitalPeriod`, `HohmannTransfer`.
- **Body-first**: `(mu, r)`, `(mu, a)`, `(mu, r1, r2)` — current `OrbitalVelocity`, what most NASA references use.

Pick one. **State-first** is more idiomatic for `reality` (parameters first, "constant" last) and matches `OrbitalPeriod`. Update `OrbitalVelocity(mu, r, a)` → `OrbitalVelocity(r, a, mu)`.

**Recommendation D3.** Optional `Body` shortcut struct for ergonomics on common bodies, *without* baking specific bodies into `orbital/`:
```go
// In a *new* package constants/celestial (or constants/bodies.go), not orbital/:
type Body struct { Mu, Radius, J2 float64; Name string }
var (
    Earth = Body{Mu: 3.986004418e14, Radius: 6.378137e6,   J2: 1.08263e-3, Name: "Earth"}
    Moon  = Body{Mu: 4.9048695e12,   Radius: 1.7374e6,     J2: 2.027e-4,  Name: "Moon"}
    Sun   = Body{Mu: 1.32712440018e20,Radius: 6.957e8,     J2: 2.2e-7,    Name: "Sun"}
    Mars  = Body{Mu: 4.282837e13,    Radius: 3.3895e6,     J2: 1.96045e-3,Name: "Mars"}
    // ... full IAU 2015 list ...
)
```
**Crucially** the `orbital/` functions still take `mu float64`, not `Body`. The `Body` struct is a **caller-side** convenience — `orbital.OrbitalPeriod(a, constants.Earth.Mu)` keeps `orbital/` body-agnostic. This separation matches `reality`'s zero-coupling principle and is what poliastro does conceptually (its `Earth` is in `bodies.py`, the propagators take `attractor`).

---

## 7. Sibling-package alignment — vector types

`physics/mechanics.go` uses scalar arguments (`NewtonSecondLaw(F, m float64)`, `ProjectilePosition(v0, theta, t, g float64)`) — **scalar-only**, never vectors. `geometry/sdf.go` and `geometry/quaternion.go` use `[3]float64` for points, `[4]float64` for quaternions — **fixed-size arrays, zero allocation**. `chaos/ode.go` uses `[]float64` slices for arbitrary-dim ODE state — **slice with allocation**. `linalg/` exposes `Vec3` (`[3]float64`) and `Vec` (`[]float64`) typed as named types.

`orbital/` is currently scalar-only (matches `physics`), but the COE↔RV transforms naturally produce 3-vectors. The cleanest alignment is:

- **Position/velocity in `orbital/`: `[3]float64`.** Matches `geometry/`. No alloc. Cross-language: JSON `[1.0, 2.0, 3.0]`.
- **State `(r,v)`: two `[3]float64` returns or one `[6]float64`.** I lean `(r, v [3]float64)` for clarity; `[6]float64` is a Vallado-textbook convention but obscures which three are position vs velocity at the call site.
- **COE: stay scalar-tuple `(a, e, i, Ω, ω, ν float64)` for now**, promote to named-type `[6]float64` once `RVtoCOE` ships and the round-trip becomes a frequent caller pattern.

**Recommendation E.** Adopt `[3]float64` for position/velocity returns from new functions (107-T1 backlog). Document in `orbital.go` package doc: *"Position and velocity vectors are `[3]float64` in the function's documented frame."*

---

## 8. Argument ordering — the ω/Ω trap

`KeplerOrbit(a, e, i, omega, capOmega, nu)` orders the angles as **i, ω, Ω, ν**. The classical-textbook ordering of Keplerian elements is **a, e, i, Ω, ω, ν** (Vallado, Bate-Mueller-White, Curtis, Roy). The current Go signature swaps Ω and ω. This is **one transposition away from a silent rotation error** — and a Python user copy-pasting an `(a,e,i,raan,argp,nu)` tuple from poliastro into `orbital.KeplerOrbit(*coe)` will get a wrong answer with no compile error.

**Recommendation F.** Fix the argument order to `KeplerOrbit(a, e, i, capOmega, omega, nu)` (Ω before ω, matching all standard textbooks). This is a v0.11.0 break worth taking — it is silently wrong for the *exact* use case (cross-language port from poliastro) the golden-file infrastructure is designed to support. Existing golden files re-encode trivially (the underlying float values are unchanged; only the Go function-signature order changes).

---

## 9. Summary of recommendations

| ID | Recommendation | Cost | Severity |
|---|---|---|---|
| A1 | Rename `KeplerOrbit` → `COEtoRV`, return both r and v | low | MED — current name misleads, velocity costs ~0 |
| A2 | Add `RVtoCOE(r, v, mu)` | ~80 LOC + golden | HIGH — every Tier-1 propagator round-trips on this |
| A3 | Keep `KeplerOrbit` as deprecated alias | low | LOW — migration ergonomics |
| B  | Defer `Frame` enum until frame transforms ship; document output frames in interim | low now | MED — silent foot-gun without it |
| C  | Defer `TimeScale` enum until propagator ships | low now | LOW — premature |
| D1 | `EscapeVelocity(mu, r)` not `(M, r)` | 1-line break | HIGH — violates package contract, 7-digit precision regression today |
| D2 | Standardise argument order (state-first OR body-first; pick one) | small breaks | MED — `OrbitalVelocity` outlier |
| D3 | Add `constants.Body` struct in `constants/`, not `orbital/` | ~40 LOC | LOW — caller-side ergonomics |
| E  | Use `[3]float64` for position/velocity in new functions | convention | MED — sibling-package alignment |
| F  | Fix `KeplerOrbit` arg order: Ω before ω | 1-line break | **HIGH** — currently silently wrong vs textbook order, cross-language port hazard |

**Top-three priority for next API touch.** F (Ω/ω order — silent wrong-result foot-gun), D1 (`EscapeVelocity` mu-vs-M — package contract violation), A2 (`RVtoCOE` — Tier-1 unblocker).

**What this package should explicitly *not* do yet.** Don't ship `Frame` enum without a transform function (B). Don't ship `TimeScale` enum without a propagator (C). Don't migrate to `OrbitState` struct until `RVtoCOE` and the universal-variable propagator (107-T1.1) lock the field set (Option 2 → defer). Premature abstraction here will bake choices that 107's Tier-1 work will need to re-break.

---

## 10. File references

- `C:/limitless/foundation/reality/orbital/orbital.go` (8 functions, all scalar-in, scalar-or-tuple-out)
- `C:/limitless/foundation/reality/orbital/orbital_test.go` (golden + behavioural tests)
- `C:/limitless/foundation/reality/orbital/testdata/orbital/*.json` (8 vector files)
- `C:/limitless/foundation/reality/constants/physics.go` (`GravitationalConst` — used only by `EscapeVelocity`; no μ_⊕, μ_☉, μ_☽ constants exist anywhere in `constants/` today — **gap**)
- `C:/limitless/foundation/reality/geometry/quaternion.go` (sibling using `[3]float64`/`[4]float64` — pattern to copy)
- `C:/limitless/foundation/reality/physics/mechanics.go` (sibling using scalars only — pattern *not* to copy for vectorial outputs)
- `C:/limitless/foundation/reality/chaos/ode.go` (sibling with `[]float64` slice state — relevant when propagator lands)

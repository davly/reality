# 064 — em-api

**Topic:** em — vector field types, complex-valued fields, units, naming, frequency convention, component-combination ergonomics.
**Audited:** `C:\limitless\foundation\reality\em\em.go` (213 LOC, 11 functions). Cross-referenced: `constants/physics.go`, `constants/units.go`, `linalg/vector.go`, `geometry/quaternion.go`, `control/transfer.go`, `signal/fft.go`, `acoustics/acoustics.go`.
**Sister reports:** 061 numerics (present-surface correctness + F-1..F-6 algorithm choices), 062 missing (170-220 primitives, ~14,200 LOC ladder), 063 sota (8 architectural decisions — D1 freq-first, D2 complex128, D3 FIT-doc-Yee-impl, D4 hand-roll adjoint, D5 single-method, D6 special/ subpackage, D7 BYO-mesh, D8 no-SIMD).

This report is about **Go signature shapes** — how the function names, parameter orders, return types, and naming idioms hang together — distinct from 061's algorithm choices (F-1..F-6), 062's primitive enumeration, and 063's high-level architectural decisions (D1..D8). Every section answers the question: when 062's primitives ship, **what must their Go signatures look like to compose with the existing reality conventions across `linalg/`, `constants/`, `signal/`, `control/`, `geometry/`?**

---

## Headline

The current `em/` API is **eleven free functions, all `(...float64) float64`, no types, no methods, no errors, no allocations, no complex numbers, no vectors, no error returns, no units types**. It is internally consistent and ergonomically clean for what it is: an EM-101 cheat-sheet over scalars. Every signature is `func F(args ...float64) float64`.

But that internal consistency **stops working** the moment any of the following lands: (a) an electric-field function returning a vector, (b) an impedance function returning `complex128`, (c) a function that can fail (FDTD CFL violation per F-2, S↔Z conversion per F-6), (d) a function with multiple return values (current loop B-axial returns `(B_z, B_r)`, Hertzian dipole returns `(E_r, E_θ, H_φ)`), (e) a function whose output is per-axis (`PoyntingVector` returns a `Vec3`). Of the 170-220 missing primitives in 062, **at least 90% need at least one of (a)-(e)**. So the present API uniformity is a happy accident of working only on scalar-return primitives that ship without errors.

This report names **eleven seam decisions** that must be settled before T1-COMPLEX / T1-FIELDS / T1-DIPOLE / T1-WAVEGUIDE / T1-ARRAY / T1-TLINES / T1-SMITH ship — not because any individual decision is hard, but because **first-ship across Go/Python/C++/C# binds golden-file shape and naming permanently**, and inconsistent shape choices pre-empt the kind of typed-discipline `linalg/` and `constants/` already enforce. Eight of the eleven cost zero LOC today (decisions only); three are 30-LOC-or-less doc adjustments.

---

## API surface today: the eleven existing signatures

```go
CoulombForce(q1, q2, r float64) float64                      // N
ElectricField(q, r float64) float64                          // V/m (magnitude only)
OhmsLaw(V, R float64) float64                                // A
PowerElectric(V, I float64) float64                          // W
ResistorsInSeries(resistances []float64) float64             // ohms
ResistorsInParallel(resistances []float64) float64           // ohms
CapacitorEnergy(C, V float64) float64                        // J
InductorEnergy(L, I float64) float64                         // J
RCTimeConstant(R, C float64) float64                         // s
ResonantFrequencyLC(L, C float64) float64                    // Hz (NOT rad/s)
// (variable, not function) coulombConst float64              // pre-computed N·m²/C²
```

All `(scalar, scalar, ...) -> scalar`. No errors. No types. No methods. No allocations. No complex. No vectors. No units. Two slice-input functions both use `[]float64` not variadic `...float64`. Naming uses **single-letter physics symbols mixed with words**: `q1, q2, r` (math style) but `OhmsLaw(V, R)` (capital from the formula) and `RCTimeConstant(R, C)` and `CapacitorEnergy(C, V)`. **Frequency is `Hz` not `ω`** (`ResonantFrequencyLC` returns Hz, not rad/s). No package-level `Frequency` or `AngularFrequency` type alias.

Implicit conventions inherited from this surface:
- **Physics symbols, not English words** for parameter names (`q`, `r`, `V`, `R`, `C`, `L`, `I`).
- **SI base units, not derived units** (radians not degrees, F/m not pF/cm, ohms not kohms, m not mm).
- **Hz not rad/s** for frequency-domain output.
- **No "Vec" or "Field" or "Phasor" suffix** — `ElectricField` is a magnitude scalar despite the name.
- **No errors returned** — IEEE-754 ±Inf/NaN is the failure mode, propagating through arithmetic.
- **No methods, no receivers** — every function is package-level free function.

These conventions are **not documented anywhere in `em.go`'s package doc** (lines 1-10 only say "numbers in, numbers out" and cite Coulomb's-constant derivation from `constants/`). This is the **single most-cited gap** — every seam decision below is implicitly constrained by these conventions, and every cross-language port must guess them.

---

## Cross-package convention audit

To avoid forking conventions when 062's primitives land, here's what `em/` must compose with:

### `constants/`
- **Naming:** `CamelCase` no underscores. `SpeedOfLight`, `VacuumPermittivity`, `VacuumPermeability`, `ElementaryCharge`, `BohrRadius`. Not `c`, `eps0`, `mu0`, `e`, `a0`. **English-word style, not single-letter.** This is the inverse of the scalar-parameter style `em/` uses.
- **Units inline in comment:** `const SpeedOfLight = 299792458.0 // m/s`. Every constant carries a `// units` trailing comment. `em/` should do the same in docstrings (it does, in prose, less consistently).
- **SI everywhere:** every physical constant in SI; unit conversions explicitly named `MetersPerFoot` etc. Pattern: `<TargetUnit>Per<SourceUnit>` (multiply by). **No floating-point units types** — bare `float64` carries the meaning.
- **No exact / inexact distinction in type:** SI 2019 exact constants and CODATA 2018 measured values share the same `const float64` shape; precision is documented in the `// Source:` comment. Reality has chosen "documentation is the carrier" over "type system is the carrier" — `em/` must match.

### `linalg/`
- **No `Vec3` type.** Vectors are `[]float64` with **caller-allocated output buffer**: `func VectorAdd(a, b, out []float64)`. Length-mismatch is a panic.
- **`CrossProduct(a, b, out []float64)` exists at `linalg/linalg_test.go:689-720` (matching `vector.go` body)** — three-element slice convention, not `[3]float64`.
- **`DotProduct(a, b []float64) float64` returns scalar** — bare slice in, scalar out, no Vec3.
- **Conclusion:** `linalg/` has standardized on `[]float64` + caller-out-buffer for vector arithmetic. **`em/` must match.** Per 062 §T1-FIELDS spec'ing `Vec3` (`[3]float64`-array): wrong shape, picks a fork from `linalg/`. Correct shape: `func ElectricFieldVec(charges []float64, positions, point []float64, out []float64)` with `out` length 3.

### `geometry/`
- **Uses `[3]float64` and `[4]float64` fixed-size arrays** for spatial primitives: `BezierCubic3D(p0, p1, p2, p3 [3]float64, t float64) [3]float64`, `QuatFromAxisAngle(axis [3]float64, angle float64) [4]float64`, `SDFSphere(p, center [3]float64, radius float64) float64`. Returns by value, not via out-buffer.
- **Conflict:** `linalg/` uses `[]float64` + out-buffer; `geometry/` uses `[3]float64` + return-by-value. **The repo has not picked a winner.**
- **`em/`'s seam decision:** when the 062 vector fields ship, `em/` must pick **one** of these conventions, and pick it consistently. Recommend `[3]float64` matching `geometry/` because: (a) EM fields are 3D-spatial primitives like SDFs, not arbitrary-dim vectors like ML embeddings; (b) `[3]float64` is a fixed-stack-allocated value type, no allocation, no length check, matches CLAUDE.md §3 "no allocations in hot paths"; (c) `linalg/`'s `[]float64` buffer convention is for arbitrary-dim BLAS-style work, not 3-vector geometry. **This is decision A1 below.**

### `control/`
- **Uses `complex128` natively at the surface.** `TransferFunction.Evaluate(s complex128) complex128`, `Poles() []complex128`. No `(real, imag)` pair API, no `RealPart`/`ImagPart` accessor — Go's native `complex128` IS the carrier.
- **Methods on a struct.** Not free functions. `tf.Evaluate(s)`, `tf.Poles()`, `tf.IsStable()`. State (numerator, denominator polynomials) lives in the struct.
- **Conflict with `em/`:** `em/` is free functions only, no structs, no methods. When 062's `Network` (S/Y/Z/ABCD parameters) and `Material` (Drude/Lorentz/Debye) land, do they get the `control/` struct-with-methods treatment or the `em/` free-function treatment?
- **Recommendation:** **Mirror `control/` for stateful primitives** — `Network` carries the (S, Z, Y, ABCD, ports, freqs) state, and `network.S2Z()` is a method. **Free functions for stateless math** — `Friis(Pt, Gt, Gr, lambda, R)` is `(...float64) float64`. **This is decision A2 below.**

### `signal/`
- **`FFT(real, imag []float64)` operates in-place on pre-allocated slices**, not on `[]complex128`. Comment says "FFT/IFFT operate in-place on pre-allocated real and imaginary slices." This is the **one place reality has explicitly chosen against `complex128`** — to match cross-language calling conventions where Python/C#/C++ FFT libraries also work on separate real/imag arrays for SIMD friendliness.
- **Conflict with `control/`:** `signal/` rejects `complex128`; `control/` embraces it. **The repo has not picked a winner here either.**
- **`em/`'s seam decision:** when T1-COMPLEX impedance lands, does it match `signal/` (pair of float64 slices) or `control/` (native `complex128`)? Per 061 §F-4: native `complex128`. Per 063 §D2: introduce `complex128` as second numeric type scoped to `em/`. **Recommend `complex128` matching `control/`** — impedance is scalar-per-frequency, not array-per-stream; `complex128` parallels `control/`'s scalar-per-pole exactly. **This is decision A3 below.**

### `acoustics/`
- **Uses `Hz` not `ω` for frequency** matching `em/`'s `ResonantFrequencyLC`. `DopplerShift(f0, vs, vr, c float64) float64`, `WaveLength(f, c float64) float64`, `AWeighting(f float64) float64` — all take Hz.
- **Single-letter parameters with prose docstring describing units** — same pattern as `em/`.
- **Free functions only, no struct, no methods.** Matches `em/`.
- **Conclusion:** `em/`'s present surface is consistent with `acoustics/`. When `em/` extends, **stay with Hz at the surface for consumer ergonomics** but document `ω = 2πf` rad/s within docstrings where formulas use ω.

---

## Eleven seam decisions

These are the API-shape questions that must be answered **before** 062's T1 primitives ship. Each is independent; each binds golden-file shape cross-language; eight cost zero LOC today.

### A1 — Vector field type: `[3]float64` not `[]float64`-buffer

**Today:** `ElectricField(q, r float64) float64` returns a magnitude scalar.
**Tomorrow (062 T1-FIELDS):** `ElectricFieldVec(...)` must return a 3-vector. Two choices:
- (i) **`[3]float64` return-by-value, matches `geometry/`** — stack-allocated, no allocation, no length check, idiomatic for 3D physics.
- (ii) **`[]float64` + caller-out-buffer matches `linalg/`** — heap-flexible but pays length-check cost and signature noise.

**Recommend (i) `[3]float64`.** Rationale: EM fields are spatial 3-vectors with fixed dimensionality, just like geometric primitives. Per `geometry/quaternion.go` precedent: `QuatRotateVec(q [4]float64, v [3]float64) [3]float64`. Same shape:

```go
func ElectricFieldVec(charges, positions []float64, point [3]float64) [3]float64
func MagneticFieldBiotSavart(currents, positions []float64, point [3]float64) [3]float64
func PoyntingVector(E, B [3]float64) [3]float64
func LorentzForce(q float64, v, B [3]float64) [3]float64
```

Where N-source inputs (`charges`, `positions`) flatten to `[]float64` of length N and 3N respectively (same convention as `geometry/curves.go`'s control-point arrays). **Document this as "EM-API-V1: `[3]float64` for spatial vectors, `[]float64` for collections of scalars" in package doc** — 6 LOC, prevents the `Vec3 vs [3]float64 vs []float64` fork at 062 first-ship.

### A2 — Struct-with-methods for stateful, free-functions for stateless

**Today:** all free functions.
**Tomorrow:** `Network` (S/Y/Z/ABCD parameters at N frequencies, M ports), `Material` (Drude/Lorentz/Debye dispersive), `Antenna` (pattern + impedance at N frequencies), `Waveguide` (mode set + cutoff frequencies) all carry state.

**Recommend struct-with-methods for stateful** (matches `control.TransferFunction`):

```go
type Network struct {
    S       []complex128 // flat row-major, len = nFreq * nPorts * nPorts
    Z0      float64      // reference impedance (typ. 50 ohm)
    Freqs   []float64    // Hz
    NPorts  int
}
func (n *Network) S2Z() *Network          // returns new Z-param network
func (n *Network) Cascade(other *Network) *Network
func (n *Network) ReturnLoss(port int) []float64

type Material interface {
    Permittivity(omega float64) complex128 // ε(ω), complex
}
type DrudeModel struct { OmegaP, Gamma float64 }
func (d *DrudeModel) Permittivity(omega float64) complex128 { ... }
```

**Free functions for stateless math:** `Friis`, `FreeSpacePathLoss`, `Directivity`, `EffectiveAperture`, `BrewsterAngle`, `SkinDepth`, `Z0Coax`, `Z0Microstrip`. These take only scalars and return scalars (or `complex128` when reactance enters); no state to carry.

**Why both:** consumer ergonomics. `Friis(Pt, Gt, Gr, lambda, R)` reads as a formula; `network.Cascade(other)` reads as an operation on a value. Picking only one forks ergonomics; picking both consistently follows `control/`-stateful + `em/`-stateless precedent. **5-LOC interface declaration in package doc fixes this seam at zero LOC of code.**

### A3 — `complex128` native, no `(real, imag)` float64 pairs

**Today:** zero `complex128` in `em/`.
**Tomorrow:** every RF function must return `complex128` (impedance, reflection coefficient, S-parameters, propagation constant γ = α + jβ, dispersive permittivity ε(ω), Fresnel coefficients).

**Recommend native `complex128` matching `control/`, not separate-arrays matching `signal/`.** Per 061 §F-4 + 063 §D2:

```go
func Impedance(R, L, C, omega float64) complex128       // R + jωL + 1/(jωC)
func ReflectionCoefficient(ZL, Z0 complex128) complex128
func VSWR(gamma complex128) float64                      // (1+|γ|)/(1-|γ|), real-valued
func PropagationConstant(R, L, G, C, omega float64) complex128  // γ = α + jβ
func FresnelRs(thetaI, n1, n2 complex128) complex128    // s-pol amplitude reflection
func DrudePermittivity(omegaP, gamma, omega float64) complex128
```

**Cross-language port:** Python `complex`, C++ `std::complex<double>`, C# `System.Numerics.Complex` are all IEEE-754 bit-exact equivalents. Golden file stores `[real, imag]` pair per scalar — trivially extends current JSON shape.

**`signal/` exception:** FFT keeps its real/imag-pair signature because (a) it's array-data not scalar, (b) the Cooley-Tukey radix-2 in-place implementation requires separate float64 arrays for SIMD-friendly cross-language ports, (c) callers can `complex(re[i], im[i])` to lift to `complex128` if needed. **No conflict** — different conventions for different shapes (scalar vs array).

### A4 — Frequency convention: `f` (Hz) at API surface, `ω` (rad/s) internal

**Today:** `ResonantFrequencyLC(L, C) -> Hz` not rad/s. Acoustics matches.
**Tomorrow:** every dispersion / waveguide / impedance function takes `omega` rad/s in the formula. Two choices:
- (i) **API takes `f` Hz**, multiplies by 2π internally. Matches today's surface, matches consumer expectations, matches scikit-rf ("freq in Hz" everywhere).
- (ii) **API takes `omega` rad/s**, raw. Matches Pozar/Balanis textbook formulas verbatim, matches `signal/` and `control/` (which take `s` complex frequency directly).

**Recommend (i) Hz at surface, document ω = 2πf inside.** Rationale: (a) consumers think in Hz, GHz; ω requires mental 2π multiplication, (b) every existing `em/` and `acoustics/` function is Hz, breaking consistency would be a per-package fork, (c) the 2π factor inside the function is one extra multiply — negligible. **Exception:** when a function takes a dimensionless ratio like `omega / omegaC` (waveguide cutoff), expose the **pre-divided ratio** as the parameter, not raw ω, so the API is "Hz everywhere or dimensionless".

```go
func RectWaveguideCutoff(m, n int, a, b float64) float64                  // returns Hz
func RectWaveguideTE10Beta(f float64, a float64, epsR, muR float64) float64  // takes Hz
func GroupVelocity(f, fc float64, c float64) float64                      // Hz, Hz, m/s
```

**Document at package level: "All frequency parameters and returns are in Hz. Internal computations use ω = 2πf rad/s; this is documented per-function where it matters."** — 3 LOC fix.

### A5 — Resistance vs. Impedance: distinct names, `R` for ohms-resistance, `Z` for ohms-impedance

**Today:** `ResistorsInSeries`, `ResistorsInParallel` — both say "Resistors", both return real `float64`. Correct for DC; insufficient for AC.
**Tomorrow:** AC complex impedance arithmetic must be distinct from DC resistance arithmetic, even though the underlying math (series=Σ, parallel=1/Σ(1/Z)) is identical.

**Recommend two parallel function families, distinguished by name not by polymorphism:**

```go
// DC, real-valued, scalar (existing)
func ResistorsInSeries(R []float64) float64
func ResistorsInParallel(R []float64) float64

// AC, complex-valued, with-Kahan (new, per 061 §F-4)
func ImpedancesInSeries(Z []complex128) complex128
func ImpedancesInParallel(Z []complex128) complex128
```

**Why not one polymorphic function:** Go has no overloading. Generics (`[T constraints.Float | constraints.Complex]`) work but break cross-language port (Python/C#/C++ have no analog). **Two functions, same body shape, different types — explicit, predictable, golden-fileable.**

**Naming consistency across the family:**
- `R` (ohms, real) for resistance, `X` (ohms, real) for reactance, `Z = R + jX` (ohms, complex) for impedance.
- `G` (siemens, real) for conductance, `B` (siemens, real) for susceptance, `Y = G + jB` (siemens, complex) for admittance.
- All match Pozar/Balanis textbook.

### A6 — Field naming: distinct `E`, `B`, `D`, `H`, `A`, `Φ`, `J`, `ρ` — match Maxwell convention

**Today:** `ElectricField` returns scalar (magnitude). No B, D, H, A, Φ, J, ρ functions.
**Tomorrow:** every Maxwell quantity must be distinct in name and type:

| Quantity | Letter | Units | Vector? | Function name |
|---|---|---|---|---|
| Electric field | E | V/m | yes | `ElectricFieldVec`, `ElectricFieldMag` |
| Magnetic flux density | B | T (= V·s/m²) | yes | `MagneticFieldVec` (returns B) |
| Electric displacement | D | C/m² | yes | `ElectricDisplacementVec` |
| Magnetic field strength | H | A/m | yes | `MagneticFieldHVec` |
| Vector potential | A | V·s/m | yes | `VectorPotentialA` |
| Electric potential | Φ (or V) | V | scalar | `ElectricPotential` |
| Current density | J | A/m² | yes | `CurrentDensityVec` |
| Charge density | ρ | C/m³ | scalar | `ChargeDensity` |

**Critical disambiguation:** "MagneticField" alone is ambiguous between B and H (textbooks differ). **Recommend reality-canonical: `MagneticFieldVec` returns `B` (tesla) by default** matching Griffiths/Jackson convention; `MagneticFieldHVec` returns H (A/m) explicitly named for the constitutive-relation case. **Document this in package doc** — 4 LOC, prevents the B-vs-H fork that has confused engineering students for 150 years.

### A7 — Magnitude vs. vector variant pairs: `<Quantity>Mag` and `<Quantity>Vec` siblings

**Today:** `ElectricField` returns magnitude (scalar) but is named without disambiguation.
**Tomorrow:** every vector-field function should have a magnitude sibling for callers who only need `|E|` and want to skip the sqrt-of-sum-of-squares:

```go
func ElectricFieldMag(q, r float64) float64                    // |E|, today's behavior, rename
func ElectricFieldVec(charges, positions []float64, point [3]float64) [3]float64
```

**Why not just return [3]float64 and let caller compute magnitude:** because (a) the magnitude case is the dominant consumer use (compute |E| at a point is a single sqrt, not three components), (b) backward compat — today's `ElectricField(q, r)` IS the scalar case, renaming to `ElectricFieldMag` keeps it, (c) magnitude golden vectors are scalar and golden-file simpler.

**Pattern:** every vector-field function ships a `<Name>Mag` magnitude sibling. **Backward-compat note:** rename `ElectricField -> ElectricFieldMag` and add `ElectricField` as deprecated alias for one release, then remove. **3-LOC rename + 1-line deprecation comment** when T1-FIELDS lands.

### A8 — Component-combination API: `[]float64` slice, not variadic

**Today:** `ResistorsInSeries(resistances []float64) float64` — slice argument, not `...float64` variadic.
**Tomorrow:** consistency across `ImpedancesInSeries`, `CapacitorsInParallel` (= sum), `InductorsInSeries` (= sum), Mutual-inductance-corrected variants.

**Recommend keep slice convention.** Rationale: (a) variadic `...float64` allocates a slice anyway under the hood for N>0 args, (b) cross-language port: Python `*args`, C++ has no variadic-of-double, C# `params double[]` — slice is the lowest common denominator, (c) callers usually have a slice already (test inputs, dynamic component lists). **Same pattern across the family:**

```go
func ResistorsInSeries(R []float64) float64
func ResistorsInParallel(R []float64) float64
func CapacitorsInSeries(C []float64) float64
func CapacitorsInParallel(C []float64) float64
func InductorsInSeries(L []float64) float64       // assumes no mutual coupling
func InductorsInParallel(L []float64) float64
func ImpedancesInSeries(Z []complex128) complex128
func ImpedancesInParallel(Z []complex128) complex128
```

**Naming exception:** the symbol `Sum` implies dimensionless arithmetic; **prefer `<Component>InSeries` and `<Component>InParallel` over `Sum<Components>` / `Combine<Components>` / `Aggregate<Components>`** — matches the physics, parallels `acoustics.SoundIntensity` style.

### A9 — Error-return shape: 062 F-2/F-6 use `(result, error)` for design-time, panic for hot-path-precondition

**Today:** zero error returns. `OhmsLaw(V, 0)` returns `+Inf`, `CoulombForce(q, q, 0)` returns `+Inf`. No `error` types in `em/`.
**Tomorrow:** per 061 §F-2 FDTD CFL violation must error-return; per 061 §F-6 S↔Z conversion at det(I-S)<1e-14 must error-return.

**Recommend two-tier error policy:**
- **`(T, error)` return for design-time / setup functions:** FDTD initialization, S↔Z conversion, eigenmode solve, mesh validation. These run once per simulation; the error-return cost is invisible.
- **`panic` for per-frame hot-path precondition violations:** exactly matching `linalg.LU` policy on singular matrix and `linalg.VectorAdd` policy on length mismatch. Hot-path Yee step that gets called 10⁶ times per simulation cannot afford error-tuple allocation; if CFL is going to be checked per-step, it MUST be a panic on violation, not error-return.
- **IEEE-754 ±Inf/NaN for arithmetic edges:** today's policy continues — `OhmsLaw(V, 0)` keeps returning `+Inf`, not erroring. Consumers handle ±Inf at the consuming site. Matches `linalg/` and `prob/` policy.

```go
// Design-time: error-return
func NewFDTD3D(grid GridSpec, dx, dt float64, source SourceSpec) (*FDTD3D, error)
func (n *Network) S2Z() (*Network, error)  // err if det(I-S) singular

// Hot-path: panic on precondition violation (CLAUDE.md §3 no-allocation policy)
func (f *FDTD3D) Step()  // panics if internal CFL invariant ever violated (programmer error)

// Arithmetic edges: IEEE-754 propagation
func OhmsLaw(V, R float64) float64  // +Inf/NaN propagation, no error
```

**Document policy in package doc.** ~10 LOC. Prevents the "every primitive picks its own error policy" fork.

### A10 — Naming case for compound entities: `Z0Coax`, `Z0Microstrip` — capital number, type after entity

**Today:** function names mix `OhmsLaw`, `RCTimeConstant`, `ResonantFrequencyLC`, `ResistorsInSeries`. The `LC` suffix is appended; `RC` is a prefix.
**Tomorrow:** transmission-line characteristic impedance Z₀ comes in many flavors: coax, microstrip, stripline, twisted-pair, parallel-wire, slotline, CPW. Naming choice:
- (i) `CoaxZ0(...)`, `MicrostripZ0(...)`, `StriplineZ0(...)` — entity-then-quantity.
- (ii) `Z0Coax(...)`, `Z0Microstrip(...)`, `Z0Stripline(...)` — quantity-then-entity.
- (iii) `CharacteristicImpedanceCoax(...)` — fully spelled.

**Recommend (i) entity-then-quantity:** matches Go-stdlib convention (`time.NewTimer`, `bytes.Buffer.Write`), matches `crypto/sha256.Sum`, matches `signal.FFT` (transform-then-suffix). Also matches `acoustics`'s `SabineRT60(V, A) float64` — entity (Sabine) + quantity-suffix (RT60).

```go
func CoaxZ0(D, d, epsR float64) float64                 // (60/√εᵣ)·ln(D/d)
func MicrostripZ0(W, h, epsR, t float64) float64        // Hammerstad-Jensen
func StriplineZ0(W, b, t, epsR float64) float64         // Cohn / IPC-2141
func TwistedPairZ0(D, d, epsR float64) float64
func CPWZ0(W, S, h, epsR float64) float64
```

**Naming consistency:** for any RF/microwave entity X with characteristic property Y, function name is `XY` not `YX`. Antenna pattern: `DipolePattern` not `PatternDipole`. Waveguide cutoff: `RectWaveguideCutoff` not `CutoffRectWaveguide`. **Match `geometry/`'s `BezierCubic3D`, `SDFSphere`, `QuatFromAxisAngle` precedent.**

### A11 — Unit suffixes in function names: avoid; use parameter docstring for units

**Today:** `RCTimeConstant` returns seconds, `ResonantFrequencyLC` returns Hz, `CapacitorEnergy` returns joules — none of these append `_s`, `_Hz`, `_J` to the function name. Acoustics matches: `SoundSpeed` not `SoundSpeed_mps`. Constants match: `SpeedOfLight` not `SpeedOfLight_mps`.
**Tomorrow:** consistency demands no `_dB`, `_dBm`, `_dBW`, `_dBi` suffixes. `EIRP_dBm` per 062 §T1-ANTENNA-METRICS sketch is **wrong shape** — should be `EIRPdBm` (no underscore, matches Go camelCase) or split into `EIRP` (returning W linear) and `EIRPdBm` (returning dBm). Recommend the split:

```go
func EIRP(Pt, Gt float64) float64       // watts (linear)
func EIRPdBm(Pt, Gt float64) float64    // dBm = 10·log10(EIRP/1mW)
func EIRPdBW(Pt, Gt float64) float64    // dBW = 10·log10(EIRP/1W)
func GainLinear(Gdb float64) float64    // 10^(G/10)
func GainDB(Glinear float64) float64    // 10·log10(G)
```

**Why split, not parameter flag:** flag-driven function (`EIRP(Pt, Gt, asLog bool)`) breaks the "deterministic golden file per name" contract — golden-test the linear case and the dB case separately, not as branches of one function.

**Underscore vs. no-underscore:** Go convention is no underscore (camelCase). `EIRP_dBm` reads as Python; `EIRPdBm` reads as Go. Recommend **no underscore** matching all 22 packages in reality today.

---

## Summary table: eleven decisions, total LOC, when to ship

| ID | Decision | LOC today | When to ship | Cost of getting wrong |
|---|---|---|---|---|
| A1 | Vector type `[3]float64` not `[]float64` | 6 (doc) | Before T1-FIELDS | Forks linalg vs geometry |
| A2 | Struct-with-methods stateful, free-fn stateless | 5 (interface decl) | Before T1-COMPLEX | Inconsistent stateful/stateless API |
| A3 | Native `complex128` not (re,im) pairs | 0 | Before T1-COMPLEX | Cross-package fork w/ control/ |
| A4 | Hz at surface, ω rad/s internal | 3 (doc) | Before T1-WAVEGUIDE | Mental 2π burden on consumer |
| A5 | R for resistance, Z for impedance — distinct fns | 0 | Before T2-S2P | Polymorphism breaks cross-lang port |
| A6 | E/B/D/H/A/Φ/J/ρ distinct named fns | 4 (doc) | Before T1-FIELDS | B-vs-H ambiguity |
| A7 | `<Name>Mag` + `<Name>Vec` siblings | 3 (rename + alias) | Before T1-FIELDS | Magnitude vs. vector caller fork |
| A8 | Slice not variadic for components | 0 | (already correct) | — |
| A9 | (T, error) design-time, panic hot-path, ±Inf arith | 10 (doc) | Before T3-FDTD | Per-primitive error-policy fork |
| A10 | Entity-then-quantity naming | 0 | Before T1-TLINES | Inconsistent name ordering |
| A11 | No unit suffixes in fn names; split linear/log | 0 | Before T1-ANTENNA-METRICS | dB-pair golden-file fork |

**Total LOC today:** ~31 LOC of package doc, all in `em.go` lines 1-30 area. **Cost to defer:** every 062 primitive after first-ship inherits the wrong shape; cross-language ports crystallize around it.

---

## Recommended bundle

**EM-API-1 (~31 LOC, ship NOW):** package-doc block in `em.go` codifying A1-A11 as the binding API conventions. Specifically:

```go
// API conventions (binding before any 062 T1 primitive ships):
//
// Vectors: 3D spatial vectors are [3]float64 by-value, matching geometry/.
//   Arbitrary-length collections are []float64, matching linalg/.
// Complex: complex128 native at the surface, matching control/. signal/'s
//   real/imag-pair convention applies only to FFT array-data.
// Frequency: Hz at API surface, ω = 2πf rad/s internal. Document where it
//   matters per-function.
// Errors: (T, error) for design-time setup; panic for hot-path precondition
//   violation; IEEE-754 ±Inf/NaN for arithmetic edges (zero division etc.).
// Naming: entity-then-quantity (CoaxZ0, RectWaveguideCutoff, DipolePattern).
//   No unit suffixes in function names except {linear, dB, dBm, dBW} variants
//   which split into separate functions (EIRP linear + EIRPdBm).
// Field letters: E (V/m), B (T), D (C/m²), H (A/m), A (V·s/m), Φ (V),
//   J (A/m²), ρ (C/m³). MagneticFieldVec returns B by default; explicit
//   MagneticFieldHVec returns H.
// Components: slice arguments not variadic. <Component>InSeries and
//   <Component>InParallel; complex variants are <Component>sInSeries with
//   complex128 input.
// Methods: stateful types (Network, Material, Antenna, Waveguide) get
//   structs with methods, matching control/. Stateless math is free
//   functions, matching today's surface and acoustics/.
```

**EM-API-2 (~5 LOC, ship before T1-COMPLEX):** Material interface declaration and stub Drude/Lorentz/Debye types as interface implementers, no method bodies yet. Locks the typing discipline at zero algorithmic cost.

**EM-API-3 (~3 LOC, ship before T1-FIELDS):** rename `ElectricField -> ElectricFieldMag`, add `ElectricField` deprecated alias. No behavior change, sets the `<Name>Mag` + `<Name>Vec` pattern.

Total: ~39 LOC. Zero algorithmic cost. All shape-binding before 062 first-ship.

---

## Non-overlap statement

- **061 em-numerics** owns numerical-correctness algorithm choices F-1..F-6 (Kahan, Carlson, companion-QR, LU-solve). 064 references those as binding for shape (e.g., `ImpedancesInParallel` Kahan-internally is F-4 algorithm; the **name** is 064's seam decision A5).
- **062 em-missing** owns primitive enumeration and tier ordering (170-220 functions, ~14,200 LOC). 064 names signature shape only — what each new function looks like, not which to add when.
- **063 em-sota** owns architectural decisions D1-D8 (freq-first, complex128 surface, FIT-doc-Yee-impl, hand-roll adjoint, single-method, `special/` subpackage, BYO-mesh, no-SIMD). 064 makes shape decisions A1-A11 *given* D1-D8; 063 picks PDE-solver-or-not, 064 picks how PDE-solver functions are named/shaped if they ship.
- **065 em-perf** will own per-call cycle counts, allocation budget, SIMD posture, FDTD inner-loop allocation invariant. 064 mentions allocation only via §A1 (`[3]float64` zero-allocation rationale) and §A9 (panic-not-error policy on hot-path) — both inseparable from shape choice.

---

## Bottom line

`em/`'s present 11-function `(...float64) float64` API is internally consistent **only because** it ships zero vectors, zero complex numbers, zero structs, zero error returns, zero alternative units. Of the 170-220 missing primitives in 062, ~90% need at least one of those features. **The shape consistency is a temporary artifact of the small surface, not an architecture.**

The eleven seam decisions A1-A11 settle the API-shape questions across vector type (`[3]float64` matching `geometry/`), complex type (`complex128` native matching `control/`), frequency convention (Hz surface / ω internal matching `acoustics/`), naming (entity-then-quantity, distinct E/B/D/H, magnitude/vector siblings, no unit suffixes in names except linear/dB splits), error policy (`(T, error)` design-time, panic hot-path, IEEE-754 arithmetic), and component-combination (slice not variadic).

**Total cost of getting it right today:** ~39 LOC of package documentation and a 3-LOC function rename. **Cost of deferring:** every 062 primitive after first-ship inherits the wrong shape; cross-language Python/C++/C# ports crystallize around it; retrofitting later breaks all 4 languages' golden files simultaneously. The shape decisions must land **before** T1-COMPLEX/T1-FIELDS/T1-TLINES first ship — there is no second chance per CLAUDE.md §1.

# 381 — meta-types-system (type system / units / manifolds for reality)

## Headline
Plain `float64` is the right canonical core for reality, but two narrow upgrades are warranted: (a) named scalar typedefs (`type Radians float64`, `type Kelvin float64`) on a short, surgically chosen API surface (orbital, acoustics, em, color, physics rotational/thermal), and (b) opaque struct types for non-Euclidean state (Quaternion, SO(3) rotation, Distribution — already exists) — without ever moving to phantom-type generic units.

## Audit of current type usage

Sampled signatures across `orbital`, `physics`, `em`, `acoustics`, `fluids`, `linalg`, `prob`, `geometry`:

- **Universal pattern:** every numeric input/output is bare `float64` or `[]float64`. No exported `Vec3`, `Quaternion`, `Matrix3`, `SO3`, `Radians`, `Kelvin`, `Joule` types exist (`Grep` for `type (Quaternion|Vec3|Matrix3|Manifold)` returns zero matches in `geometry/` — the package has no exported `type` decls at all).
- **`linalg`:** matrices are `[]float64` row-major + `n int` shape parameter passed alongside (`MatMul(A []float64, aRows, aCols int, B []float64, bCols int, out []float64)`). No rank/shape in the type. Caller-supplied output buffers (correctly — hot-path constraint).
- **`orbital.KeplerOrbit(a, e, i, omega, capOmega, nu float64)`:** mixes a length (semi-major axis in m), a dimensionless eccentricity, and four angles (radians) in one positional arg list. Comments document units; the type system does not. Misordering `(a, e)` ↔ `(e, a)` compiles silently.
- **`physics.IdealGas(n, T, V float64)`:** mol, K, m³ — same shape risk. `physics.NewtonCooling(h, A, Ts, Tinf float64)` mixes W/(m²·K), m², K, K — `Ts` and `Tinf` are interchangeable at the type level even though `h` and `A` are not.
- **`em.OhmsLaw(V, R)` / `acoustics.DopplerShift(f0, vs, vr, c)`:** four positional `float64`s of three different physical kinds (Hz, m/s, m/s, m/s) — easy to put `c` in the `vs` slot.
- **`prob.Distribution` interface (`PDF`, `CDF`)** already exists and is the right shape — analogous to statrs's `Continuous`/`ContinuousCDF`. No `Discrete` split today; `BetaDist`/`NormalDist`/`ExponentialDist`/`UniformDist` all implement the unified interface. Cross-language partner languages (Haskell `Distribution`, C# `IDistribution`) converged on the same shape per the file's own comment.
- **`constants/units.go`:** has `RadiansToDegrees`, `DegreesToRadians`, `MetersPerFoot`, etc. — purely scalar conversion constants, not types. Conversion is the user's manual responsibility.

**Bug-likelihood ranking (where unit/dim confusion bites hardest):**

1. **`orbital`** — angles in radians vs degrees, GM (`mu`) vs M (mass), m vs km. Mars Climate Orbiter risk class.
2. **`physics` thermo** — K vs °C is the perennial silent failure (`StefanBoltzmann(T,...)` requires K; `NewtonCooling(Ts, Tinf)` also K).
3. **`acoustics`** — Hz vs angular frequency ω (rad/s), Pa vs dB.
4. **`em`** — V/m vs V, Ω vs kΩ in `ResistorsInSeries([]float64)`.
5. **`color`** — already partly mitigated by separate space types in docs; sRGB vs linear-RGB is the canonical trap and not type-enforced.
6. **`linalg`** — shape-mismatch errors (passing `aCols ≠ bRows`) currently surface as runtime panics or silent garbage; rank/shape types would catch at compile time but at huge API cost.

## Options analysis

### 1. Keep plain `float64` everywhere (status quo)

- **Pros:** Zero generics tax. Trivial cross-language port — Python `float`, C++ `double`, C# `double` map 1:1. Golden-file vectors stay JSON `number`. No allocation. No API surface explosion. Aligns with Design Rule 3 ("no allocations in hot paths") and Rule 6 ("reimplement from first principles" — minimal abstraction).
- **Cons:** Mars Climate Orbiter risk class permanently latent. Documentation-only unit contract. Caller has to remember degree-vs-radian for every angle parameter.
- **Verdict:** Right default for ≥80% of the library (linalg, calculus, signal, optim, prob math, graph weights, info theory). Wrong for the ~20% where physical units are the function's whole point.

### 2. Phantom types via empty-struct tag parameters (Boost.Units / `uom` / F# UoM analog)

```go
type Quantity[D Dim] struct { v float64 }
type Length struct{}; type Mass struct{}; type Time struct{}
// L * M / T^2 = Force...
```

- **Pros:** Compile-time dimension checking; Boost.Units / Rust `uom` / F# UoM all demonstrate zero runtime overhead. Mars Climate Orbiter eliminated.
- **Cons specific to Go and to reality:**
  - Go generics (1.18+, mature in 1.24/1.25 with generic type aliases) **cannot express type-level integer arithmetic on exponents**. F# UoM, Rust `uom`, and Boost.Units all rely on either type-level numerals (Boost MPL, Rust typenum) or compiler-special-cased syntax (F# `[<Measure>] type m`). Go has neither. Encoding `kg·m/s²` requires hand-written `Force struct{}` types, which means a combinatorial explosion of derived-quantity types — F#'s killer feature is automatic derivation; Go can't do it without `go generate`.
  - `Quantity[Length].Add(Quantity[Mass])` becomes a non-error at the type level if both reduce to a `Quantity[T any]`-shaped interface — Go's structural type system fights this.
  - **Cross-language portability shatters.** Golden-file vectors are JSON numbers; if the Go API takes `Quantity[Length]`, the Python/C++/C# validators must either adopt their own UoM systems (different in each) or strip back to floats — defeating the parity story. CLAUDE.md explicitly lists "cross-language portability" as a project pillar (4 languages, golden-file driven).
  - `aicore` (the consumer) currently passes `float64`; every consumer call site changes.
- **Verdict:** Reject. The cost-benefit is upside-down for a library whose central design constraint is 4-language golden-file parity.

### 3. Struct wrappers — named scalar typedefs (`type Radians float64`)

```go
type Radians float64
type Kelvin  float64
type Meters  float64
func (r Radians) Degrees() float64 { return float64(r) * 180/math.Pi }

func KeplerOrbit(a Meters, e float64, i, omega, capOmega, nu Radians) (x, y, z Meters)
```

- **Pros:**
  - Compile-time catch of `Radians` vs `Degrees` confusion **and** of swapping a `Kelvin` arg into a `Meters` slot.
  - **Zero runtime cost** — `type T float64` is a Go alias-with-identity, identical machine code.
  - **Cross-language transparent** — JSON serialization is still `number`. Python/C++/C# validators see `double`. The Go-side type is invisible to golden-file infrastructure.
  - No generics. Works on Go 1.0. No API explosion: ~10 named scalars (`Radians`, `Degrees`, `Meters`, `Kilograms`, `Seconds`, `Kelvin`, `Joules`, `Pascals`, `Newtons`, `Hertz`) cover ≥90% of the physics-dimensional surface.
  - Composable with existing `[]float64` — `[]Radians` is a distinct type but byte-identical layout.
- **Cons:**
  - Does **not** prevent `Kelvin + Meters` arithmetic — Go allows `+`/`-` between identically-underlying-typed values across distinct names only with explicit conversion, but `Radians + Radians` requires the user to know which is which (still a win over status quo).
  - Caller must explicitly convert: `KeplerOrbit(Meters(7e6), 0.01, Radians(0), ...)`. This is the F# UoM "decoration" friction also flagged in fslang-suggestions #892.
  - Doesn't catch derived-unit errors (force = mass × accel) — only surface-level slot confusion.
- **Verdict:** **Adopt narrowly.** Best ratio of safety to cost.

### 4. Generic constraints (`Quantity[T constraint.Float]`)

- **Pros:** Lets `float32`/`float64` polymorphism (some hot-path renders, e.g., Pistachio at 60 FPS, prefer `float32`).
- **Cons:** Reality is float64-only by design (see CLAUDE.md: 256-bit big.Float for golden generation, float64 runtime). Generic numeric is solving a problem reality has chosen not to have. Adds Go 1.18+ requirement transitively to all consumers.
- **Verdict:** Reject for units; accept only if a separate float32 SIMD path becomes a real requirement (it is not today).

### 5. Manifold types (Quaternion, SO(3), Stiefel)

- **Pros:** Quaternion-as-`[4]float64` is currently a convention; making it `type Quaternion [4]float64` with methods (`Normalize`, `Mul`, `Slerp`, `ToMat3`) prevents `[]float64` of length 3, 4, or 16 from being silently passed. SO(3) ≠ R^9: a 9-element array can be a non-rotation; a `Rotation3` opaque type guarantees the invariant `R^T R = I` at every entry point.
- **Cons:** Adds allocations if returned by value frequently — but `[4]float64` is stack-allocated, so no GC. Cross-language: JSON arrays of length 4/9 still serialize cleanly; the Go-side wrapper is invisible in goldens.
- **Verdict:** **Adopt for `geometry`** (Quaternion, Rotation3, AffineTransform). Defer Stiefel/Grassmann/general manifold abstractions — Manopt-style typeclass hierarchies are overkill until reality has a real Riemannian-optimization consumer.

### 6. Tensor rank/shape types (TensorFlow / PyTorch style)

- **Pros:** Catches `MatMul(3×4, 5×6)` at compile time.
- **Cons:** Requires either dependent types (Go has none) or runtime shape checks (already what the current `aRows, aCols, bRows` ints provide). Hasktorch / TF compile-time shapes use type-level naturals; Go can't.
- **Verdict:** Reject. Current `(A, rows, cols)` triple is fine; it documents shape and panics on mismatch at runtime, which is the same place a typed-shape mismatch would fire in practice.

### 7. Error vs uncertainty wrappers (`type Measured struct { Mean, Sigma float64 }`)

- **Pros:** Real measurement chains (sensor → calibration → derived) want propagation of σ. CODATA constants in `constants/` already have nominal-value-plus-uncertainty pairs in the source comments.
- **Cons:** Touches every function — `Add(Measured, Measured) Measured` must be paired with `Add(float64, float64) float64`, doubling the API. Most reality consumers don't track σ.
- **Verdict:** **Add as an opt-in `prob/uncertainty` subpackage** with `type Measured struct { Mean, Sigma float64; Cov float64 }` and a small set of propagation helpers (linear and Monte Carlo). Do **not** thread it through the main APIs.

## Recommendation

**Adopt a three-tier type-system policy.** Articulate it in `ARCHITECTURE.md` and apply it package by package.

### Tier 1 — Plain `float64`/`[]float64` (default; ~80% of API)
`linalg`, `calculus`, `signal`, `optim`, `prob` (math layer), `graph`, `info`, `crypto`, `compression`, `combinatorics`, `chaos`, `gametheory`, `queue`, `control`, `topology`, `infogeo`, `sequence`, `changepoint`, `timeseries`, `audio` numeric kernels, `autodiff`. Stay as-is. These are mathematically dimensionless.

### Tier 2 — Named scalar typedefs (~15% of API: physics-dimensional)
Add to `constants` (or new `constants/units` types-only file):
```go
type Radians float64
type Degrees float64
type Meters  float64
type Kilograms float64
type Seconds float64
type Kelvin float64
type Pascals float64
type Newtons float64
type Joules  float64
type Hertz   float64
```
Plus `(Radians).Deg() Degrees`, `(Degrees).Rad() Radians`, `(Kelvin).Celsius() float64` etc.

Apply selectively to highest-bug-likelihood APIs:
- `orbital.KeplerOrbit`, `OrbitalPeriod`, `EscapeVelocity`, `HillSphere`, `TrueAnomalyFromMean` — angles → `Radians`, distances → `Meters`, masses → `Kilograms`.
- `physics/thermo.go` — temperatures → `Kelvin`.
- `acoustics` — frequency-bearing args → `Hertz`, `c`/`vs`/`vr` → distinct (no `MetersPerSecond` needed; just docstring).
- `em.OhmsLaw`, capacitor/inductor — voltage/current can stay `float64` (low confusion), but `RCTimeConstant` returns `Seconds`.

Skip `fluids`, `physics/mechanics`, `physics/materials`, `physics/optics` for now — many composite-unit signatures (W/(m²·K)) where naming explodes. Revisit if a real bug shows up.

### Tier 3 — Opaque structural types (~5% of API)
- `geometry.Quaternion = [4]float64` with methods.
- `geometry.Rotation3` (SO(3) invariant-preserving wrapper around `[9]float64`).
- `prob.Distribution` already exists — keep, optionally split into `Distribution` and `DiscreteDistribution` mirroring statrs `Continuous`/`Discrete`.
- New `prob/uncertainty.Measured{Mean, Sigma}` as opt-in subpackage; do **not** retrofit core APIs.

### Reject outright
- Phantom-typed `Quantity[Dim]` à la Boost.Units / Rust `uom` / F# UoM. Generics tax + cross-language parity loss exceed benefit for a 4-language golden-file library.
- Tensor rank/shape types. Go's type system can't express them; runtime checks suffice.
- Generic numeric over `float32 | float64`. Reality is float64-canonical by design.

### Migration tactic (low-cost rollout)
1. Land tier-2 typedefs in `constants` as additive change (no signature changes yet).
2. Per-package PRs: switch one high-risk function family at a time (`orbital` first — highest bug-likelihood).
3. Cross-language validators read JSON numbers; no change to Python/C++/C# golden harness.
4. Migration docstring: "Callers passing `float64` literals must wrap: `Radians(0.5)`. Use `constants.DegreesToRadians * 30` then wrap, or use the new `Degrees(30).Rad()`."
5. Defer composite/derived-unit types unless a concrete bug report justifies them.

### Why not full UoM
The library's central design constraint (CLAUDE.md, "Golden files are the proof", 4-language parity) directly conflicts with phantom-type generic units. F# UoM works because F# is the consumer language; Boost.Units works inside C++ codebases that don't cross language boundaries. Reality is the opposite: a polyglot truth library where the type system must remain quiet at the wire boundary. Tier-2 named scalars are quiet at the wire (JSON `number`), loud at the call site — exactly the trade-off this library should make.

## Sources

- [Units of Measure — Microsoft Learn (F#)](https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/units-of-measure)
- [Types for Units-of-Measure: Theory and Practice — Andrew Kennedy](http://typesatwork.imm.dtu.dk/material/TaW_Paper_TypesAtWork_Kennedy.pdf)
- [Ease conversion between Units of Measure and undecorated numerals — fsharp/fslang-suggestions #892](https://github.com/fsharp/fslang-suggestions/issues/892)
- [uom — Rust units crate (docs.rs)](https://docs.rs/uom/latest/uom/)
- [iliekturtles/uom — GitHub](https://github.com/iliekturtles/uom)
- [Boost.Units 1.82 — Boost C++](https://www.boost.org/doc/libs/1_82_0/doc/html/boost_units/Units.html)
- [Compile-time numerical unit dimension checking in C++11 — Benjamin Jurke](https://benjaminjurke.com/content/articles/2015/compile-time-numerical-unit-dimension-checking/)
- [nholthaus/units — header-only C++14 dimensional analysis](https://github.com/nholthaus/units)
- [Manopt.jl — SO(n) manifold types](https://manoptjl.org/v0.1/manifolds/rotations/)
- [Minimization on the Lie Group SO(3) and Related Manifolds — Taylor (UPenn)](https://www.cis.upenn.edu/~cjtaylor/PUBLICATIONS/pdfs/TaylorTR94b.pdf)
- [statrs::distribution::Continuous — docs.rs](https://docs.rs/statrs/latest/statrs/distribution/trait.Continuous.html)
- [statrs::distribution::Discrete — docs.rs](https://docs.rs/statrs/latest/statrs/distribution/trait.Discrete.html)
- [Go 1.24 generic type aliases — InfoQ](https://www.infoq.com/news/2025/02/go-1-24-generic-aliases/)
- [Go 1.25 highlights, generics maturity — DEV.to](https://dev.to/leapcell/go-125-highlights-how-generics-and-performance-define-the-future-of-go-4pdh)
- [Dimensional analysis in Rust: how not to crash Mars Climate Orbiter — nodraak.fr](https://blog.nodraak.fr/2021/03/dimensional-analysis-in-rust/)

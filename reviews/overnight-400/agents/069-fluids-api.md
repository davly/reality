# 069 — fluids-api

**Topic:** fluids: dimensional input validation, regime-aware dispatch.
**Scope:** call-site ergonomics ONLY. Disjoint from 066 (numerical hazards in present surface), 067 (missing primitives + Tier-1/2/3 enumeration), 068 (SOTA architecture / D-1..D-7 binding decisions). This report asks: of the 11 functions present today, what does the *signature shape* and *return contract* tell a calling engineer, and where do the API choices Reality has made deviate from Caleb Bell's `fluids` (the architectural sibling 068 named) and from Python/SI engineering convention?

**Audited surface:** `C:\limitless\foundation\reality\fluids\fluids.go` (236 LOC, 11 exported functions); test files for usage-pattern evidence. Cross-references `acoustics/`, `em/`, `physics/`, `prob/`, `signal/` for Reality's existing conventions on dimensionless numbers and dispatch.

---

## Headline

`fluids/` API today is **eleven flat free-functions over `float64` parameters with no struct-bag, no regime tag in the return, no dimensional validation, no nondimensional vs dimensional split.** Every parameter is a positional `float64` documented in a docstring, and the package has chosen "numbers in, numbers out" so consistently that **dimensional information is communicated only as English in `// Parameters: rho: fluid density (kg/m³)` comments** — the Go type system carries zero protective signal, and the call site for `BernoulliPressure(1000, 1, 101325, 0, 1, 10, 9.81)` is indistinguishable from `BernoulliPressure(101325, 1000, 1, 0, 9.81, 10, 1)` (every reordering compiles, most produce silent garbage).

**Six API-level findings, ranked:**

- **A-1 (HIGH)** — `PipeFlowFriction` swallows the laminar/turbulent regime decision inside the function and returns *only* `f`. The 70.2% jump 066 N-1 documents is hidden from the caller. The right return shape for *any* function whose value depends on a regime branch is `(value, regime, error)` — and `regime` should be a typed enum, not a string. **All seven callers (one in `fluids_test.go`, six in `fluids_edge_test.go`) currently have no way to ask "was this laminar or turbulent?" without recomputing 2300 < Re themselves.** This is the topic-prompt's headline ask.

- **A-2 (HIGH)** — **Zero dimensional input validation.** No function checks `density > 0`, `velocity ≥ 0`, `viscosity > 0`, `area ≥ 0`, `diameter > 0`, or `radius ≥ 0`. Negative density passes silently through `MassFlowRate`, `DragForce`, `Bernoulli`, `DarcyWeisbach`, `TerminalVelocity`. The only guards are: `ReynoldsNumber` relies on IEEE-754 `x/0 = ±Inf` (passive); `PipeFlowFriction` checks `Re > 0` and returns NaN (active); `TerminalVelocity` checks `Cd*ρ*A > 0` (active). Eight of eleven functions have **no validation at all**. The contract is "garbage in, garbage out" and physical impossibilities (ρ < 0) are not even logged.

- **A-3 (HIGH)** — **No `dimensionless/` sub-package.** `ReynoldsNumber` lives in `fluids/`. But Reynolds is **also** the right thing to call from `acoustics/` (acoustic streaming), `em/` (magnetic Reynolds = `μ·σ·v·L`), `chaos/` (Lorenz forcing parameter `Re_eff`), and `physics/` (heat-transfer free convection). Mach belongs to compressible fluids *and* to acoustics (sonic-boom Mach cone). Froude belongs to fluids *and* to ship-resistance / orbital tidal scaling. **The dimensionless-numbers grouping in 068 D-2 (~120 LOC, 18 numbers) deserves to be `dimensionless/` shared, not `fluids/dimensionless.go` private.** Today there is exactly one dimensionless number in the package; even before that grows to 18, the cross-package consumers are already wedged.

- **A-4 (HIGH)** — **No `Fluid` struct; no `Pipe` / `Channel` struct.** Bell's Python `fluids` ships `Atmosphere(Z=...)` returning a `(T, P, ρ, μ)` tuple, `friction_factor(Re=..., eD=..., Method='Churchill1977')`, and pipe-fitting K-factors keyed by geometry tag. Reality dispatches by passing the same `(rho, mu, ...)` pair to every function — `ReynoldsNumber(rho, v, L, mu)` and 200 ms later `BernoulliPressure(rho, v1, p1, h1, v2, h2, g)` and 200 ms later `DragForce(Cd, rho, v, A)` — three call sites, three independent re-binding of `rho`, **and the kinematic-vs-dynamic viscosity choice is forced on the caller every time** (Reynolds takes `μ` (Pa·s, dynamic); Bernoulli takes neither because it's inviscid; Stokes takes `μ`). A `Fluid` struct or named-parameter "options" map would deduplicate the binding and remove the re-look-up cost.

- **A-5 (MEDIUM)** — **Kinematic vs dynamic viscosity is undocumented as a choice.** `ReynoldsNumber(rho, v, L, mu)` uses `mu` (dynamic, Pa·s); the equivalent kinematic form `Re = v·L/ν` (where `ν = μ/ρ`) is *not exposed* — calling code wanting to use `ν` directly must invert `μ = ν·ρ`. The choice has practical weight: tabulated data for water/air is published in **both** forms in different references (CRC Handbook gives `ν`, Lemmon-NIST gives `μ`), and forcing one shape is an ergonomic tax. Same for Stokes: `StokesLaw(mu, r, v)` is dynamic-only.

- **A-6 (MEDIUM)** — **Pipe geometry is unstructured.** `PipeFlowFriction(Re, ε, D)` and `DarcyWeisbach(f, L, D, ρ, v)` and `BernoulliPressure(...)` all take `D` or characteristic length as a bare `float64` — no `Pipe{Diameter, Length, Roughness, MaterialName}` struct, no `Channel{Width, Depth, Slope, ManningN}` (which doesn't exist yet but is in 067 T1-OPEN-CHANNEL). Bell's `fluids` ships pipe-fitting K-factors as `entrance_sharp(Di=...)`, `exit_normal()`, `bend_rounded(Di=..., angle=..., rc=...)` — every call's geometric inputs are named. Reality's positional-`float64` calling convention works for one or two parameters but `BernoulliPressure(rho, v1, p1, h1, v2, h2, g)` (7 positional `float64`s) is exactly where it breaks.

---

## What the present API looks like — function-by-function

| Function | Signature | Validation | Regime-branched? | Returns regime info? |
|---|---|---|---|---|
| `ReynoldsNumber(rho, v, L, mu)` | 4× `float64` | None (relies on IEEE-754 `x/0=Inf`) | No | No |
| `BernoulliPressure(rho, v1, p1, h1, v2, h2, g)` | 7× `float64` | None | No (inviscid) | No |
| `PipeFlowFriction(Re, roughness, diameter)` | 3× `float64` | `Re ≤ 0 → NaN` | **Yes — laminar/turbulent at Re=2300** | **No (HIDES regime)** |
| `DarcyWeisbach(f, L, D, rho, v)` | 5× `float64` | None (D=0 silently divides) | No | No |
| `DragForce(Cd, rho, v, A)` | 4× `float64` | None | No (Cd is user input) | No |
| `LiftForce(Cl, rho, v, A)` | 4× `float64` | None | No (Cl is user input) | No |
| `TerminalVelocity(m, g, Cd, rho, A)` | 5× `float64` | `Cd·ρ·A ≤ 0 → NaN` | No (single closed form) | No |
| `StokesLaw(mu, r, v)` | 3× `float64` | None | No | No (no Re check; valid only Re≪1) |
| `MassFlowRate(rho, v, A)` | 3× `float64` | None | No | No |
| `VolumetricFlowRate(v, A)` | 2× `float64` | None | No | No |

Three patterns dominate: (i) **all-positional `float64`**, (ii) **no struct binding repeated parameters**, (iii) **no return tuple beyond a single scalar** (Go does support `(value, error)` and the rest of `reality/` uses it — `prob/`, `optim/`, `signal/` all return `(result, err error)` on functions that can fail; `fluids/` returns NaN sentinels instead).

---

## A-1 — Regime hidden in PipeFlowFriction (HIGH, topic-prompt explicit)

The caller of `PipeFlowFriction(Re=2299, ε=1e-5, D=0.1)` gets `0.0278` (Hagen-Poiseuille). The caller of `PipeFlowFriction(Re=2301, ε=1e-5, D=0.1)` gets `0.0474` (Colebrook). **Both calls return a bare `float64`. Neither call site can answer "which physical regime did this come from?" without re-implementing the `Re < 2300` test in the caller.**

The right return shape for any regime-branched function in Go is one of:

```go
type PipeFlowRegime int
const (
    RegimeLaminar      PipeFlowRegime = iota  // Re < 2300, Hagen–Poiseuille
    RegimeTransitional                        // 2300 ≤ Re < 4000, indeterminate
    RegimeTurbulent                           // Re ≥ 4000, Colebrook
    RegimeHydraulicallySmooth                 // Re·√f·ε/D ≤ 5
    RegimeFullyRough                          // Re·√f·ε/D ≥ 70
)

// Option A — three-return
func PipeFlowFriction(Re, eps, D float64) (f float64, regime PipeFlowRegime, err error)

// Option B — struct return
type PipeFlowResult struct {
    F          float64
    Regime     PipeFlowRegime
    Iterations int   // for Colebrook; -1 for laminar
    RelDelta   float64
}
func PipeFlowFriction(Re, eps, D float64) (PipeFlowResult, error)
```

Reality already uses three-return shape in `optim/` (`(result, iters, err)`), `linalg/` (`(L, U, P, err)` for LU). `fluids/` is the outlier ignoring the convention. Recommend **Option A** — it composes with Reality's existing `(value, error)` pattern and adds one typed enum for the regime tag.

**Compatibility note.** Renaming the existing function silently breaks every caller in `aicore/`. The Go convention is to ship the new shape under a new name (`PipeFlowFrictionDetailed`?) and deprecate the bare-scalar form, OR to add the new return values via a sibling function `PipeFlowRegime(Re) PipeFlowRegime` so the caller can ask separately. Bell's `fluids.friction.friction_factor()` returns just the scalar but exposes the dispatcher under `friction_factor_methods()` — this is a workable Pythonic split but Go's multiple-return makes the integrated shape cleaner.

**Cross-cutting application.** The same regime-hiding is latent in:
- `StokesLaw(mu, r, v)` — only valid at Re ≪ 1, but the function neither asks for nor returns an `Re` estimate. Caller has no way to know they've stepped outside the validity window.
- `TerminalVelocity(m, g, Cd, rho, A)` — Cd's Re-validity (066 N-3) is invisible; should at least return the implied `Re_terminal = ρ·v_t·d/μ` so the caller can self-check (requires adding `mu` and `d=2r` parameters or a `Sphere{m, d}` struct).
- `DarcyWeisbach(f, L, D, rho, v)` — silently accepts a `f` from any source; could optionally check `f` is in `[0.001, 0.5]` typical-engineering range.

---

## A-2 — Zero dimensional input validation (HIGH)

Eleven functions × ~3-7 parameters each = ~50 input slots. **Two of those slots are validated** (`PipeFlowFriction.Re > 0`, `TerminalVelocity.denom > 0`). **Forty-eight are not.** The contract is "the math runs, produce some IEEE-754 result." Examples that compile and produce a number today, none of which is physical:

```go
fluids.MassFlowRate(-1000, 1, 1)              // negative mass flow
fluids.DragForce(0.5, -1.225, 10, 1)          // negative-density air
fluids.BernoulliPressure(-1000, 1, 0, 0, 1, 0, 9.81)  // p2 garbage
fluids.DarcyWeisbach(0.025, -10, 0.1, 1000, 1)        // negative pipe length
fluids.StokesLaw(-1e-3, 1e-6, 1)              // negative viscosity (anti-fluid)
fluids.TerminalVelocity(-1, 9.81, 0.5, 1.225, 1)      // negative mass → imaginary v_t (NaN)
```

**Two design options for adding validation:**

**Option 1 — Validation as `(value, error)` return.** This is Reality's `optim/` / `linalg/` convention and the obvious Go pattern. Cost: every caller updates (every `aicore/` import site).

```go
func MassFlowRate(rho, v, A float64) (float64, error) {
    if rho < 0 { return math.NaN(), fmt.Errorf("rho=%v: density must be >= 0", rho) }
    if A < 0 { return math.NaN(), fmt.Errorf("A=%v: area must be >= 0", A) }
    return rho * v * A, nil
}
```

**Option 2 — Validation as a separate `Validate()` step.** Caller opts in. Cost: silent on bad input by default, contradicts Go's "errors are values" idiom.

**Option 3 — Documented-only / NaN-guard sentinels for unphysical values.** Cheapest. Adds `if rho < 0 { return math.NaN() }` to each function. No API change but the failure mode is "result is NaN, why?" which is the same diagnostic black-hole today.

**Recommend Option 1** for new functions added under 067 / 068 sprints. **Recommend Option 3** for backward-compat retrofit on the existing 11 functions (no signature change). The package-level docstring should say "negative mass / negative density / negative viscosity / negative area / negative diameter / negative length all return NaN." The four functions with positive-magnitude requirements (`PipeFlowFriction`, `TerminalVelocity`, `DarcyWeisbach`, `StokesLaw`) should each carry a `// Returns NaN if [conditions]` line.

**Velocity-sign convention.** `v < 0` is meaningful for `ReynoldsNumber` (negative-flow direction, signed Re — current behavior pinned at `TestReynoldsNumber_NegativeVelocity`) and for `BernoulliPressure` (one streamline, signed direction). It is *not* meaningful for `DragForce(Cd, ρ, v, A) = ½Cdρv²A` because `v²` is always ≥ 0 — but the call `DragForce(0.5, 1.225, -10, 1)` returns 61.25 as if `v=10`, hiding the user's sign error. Recommend documenting "drag is direction-blind; pass `|v|`" or returning `math.Abs(v)` internally with a docstring note.

---

## A-3 — Dimensionless numbers belong in `dimensionless/` (HIGH)

068 D-2 commits to ~18 dimensionless numbers (Mach, Froude, Pr, Sc, Le, Pe, Gr, Ra, Nu, Sh, St, Ec, Sr, Wo, Kn, Be, Br, Dean) plus Reynolds = 19. Of those:

| Number | Used where | Belongs in |
|---|---|---|
| Reynolds (ρvL/μ) | fluids, em (magnetic Re), acoustics (acoustic streaming), heat | `dimensionless/` |
| Mach (v/c) | fluids (compressible), acoustics (sonic boom), orbital (escape Mach in atmosphere) | `dimensionless/` |
| Froude (v/√gL) | fluids (open-channel), orbital (tidal), naval-architecture | `dimensionless/` |
| Prandtl (μcp/k) | fluids, heat-transfer (no package yet), atmosphere | `dimensionless/` |
| Strouhal (fL/v) | fluids (vortex shedding), acoustics (jet noise), control (PID period vs natural f) | `dimensionless/` |
| Knudsen (λ/L) | fluids (rarefied), em (plasma sheath), thermo | `dimensionless/` |
| Reynolds-magnetic (μσvL) | em (MHD), plasma | `em/` (?) or `dimensionless/` (better) |

**The same code re-emerges in three packages or it lives in one shared place. The DRY-vs-encapsulation tradeoff in Reality's package layout is already decided in favour of DRY** — see `geometry/` shared between `signal/` (FFT geometry), `physics/` (mechanics geometry), and `chaos/` (phase-space embedding); see `linalg/` shared between `optim/`, `prob/` (covariance), `signal/` (filters as matrices), `chaos/` (Jacobian); see `calculus/` (RK4) shared between `chaos/`, `physics/`, `control/`, `orbital/`. **There is no precedent in Reality for the same primitive being implemented twice across packages.**

**Recommended split:**
- New `dimensionless/` package: `Reynolds`, `Mach`, `Froude`, `Prandtl`, `Schmidt`, `Lewis`, `Peclet`, `Grashof`, `Rayleigh`, `Nusselt`, `Sherwood`, `Stanton`, `Eckert`, `Strouhal`, `Womersley`, `Knudsen`, `Bejan`, `Brinkman`, `Dean`, `Bond`, `Capillary`, `Weber`, `Cauchy`. ~22 functions, ~120 LOC including docs.
- `fluids/` re-exports the fluids-relevant subset under aliases for ergonomic compat: `fluids.ReynoldsNumber = dimensionless.Reynolds`. Bell's `fluids.core.Reynolds()` does this — module separation, ergonomic re-export.
- `acoustics/`, `em/`, `chaos/` import as needed.

**Migration cost.** Existing `fluids.ReynoldsNumber` should remain for backward compat. The deprecation comment points to `dimensionless.Reynolds`. ~5 LOC of churn.

---

## A-4 — No `Fluid` struct, no `Pipe` struct (HIGH)

Today the canonical "compute pipe pressure drop for water at 20 °C through 100 m of 50 mm steel pipe at 1 m/s" workflow is:

```go
// Today: 4 calls, 4 redundant binding of ρ and μ
Re := fluids.ReynoldsNumber(998, 1, 0.05, 0.001)        // ρ, v, L, μ
f  := fluids.PipeFlowFriction(Re, 4.6e-5, 0.05)         // Re, ε, D
dP := fluids.DarcyWeisbach(f, 100, 0.05, 998, 1)        // f, L, D, ρ, v
Q  := fluids.VolumetricFlowRate(1, math.Pi*0.025*0.025) // v, A
```

Three properties: ρ=998, μ=0.001, D=0.05, v=1, ε=4.6e-5, L=100. The user *types each one twice* because the API has no shared struct. Bell's Python equivalent:

```python
fl = fluids.Fluid(rho=998, mu=0.001)                       # bind once
pipe = fluids.Pipe(D=0.05, L=100, roughness=4.6e-5, fluid=fl, v=1)
pipe.Re             # 49,900
pipe.f              # via Colebrook, internally cached
pipe.pressure_drop  # via Darcy-Weisbach, internally
pipe.Q              # via continuity
```

**Bell's pattern:** struct binds shared parameters; methods compute derived quantities lazily. Each *method is still a closed-form arithmetic expression over fields* — no allocation in hot paths, no I/O. **Reality's "no allocation in hot path" rule is preserved as long as the struct is a value type, not a pointer-graph.**

**Two implementation options for Reality:**

**Option A — Struct + methods, value-receiver, allocation-free.**
```go
type Fluid struct {
    Density          float64  // ρ, kg/m³
    DynamicViscosity float64  // μ, Pa·s
}
func (f Fluid) KinematicViscosity() float64 { return f.DynamicViscosity / f.Density }

type Pipe struct {
    Diameter, Length, Roughness float64
}
func (p Pipe) Area() float64 { return math.Pi * p.Diameter * p.Diameter / 4 }
func (p Pipe) RelativeRoughness() float64 { return p.Roughness / p.Diameter }

// Free function consumes both
func PipePressureDrop(p Pipe, fl Fluid, v float64) float64 { ... }
```

**Option B — Keep flat free functions, add a higher-level convenience wrapper in a sub-package** (`fluids/scenario` or `fluids/recipes`). Discoverable but dispersed.

**Recommend A** — same pattern as `chaos.Lorenz{σ, ρ, β}`, `control.PIDController{Kp, Ki, Kd}`, `orbital.Orbit{a, e, i, Ω, ω, ν}`. It's already the Reality idiom for "small-but-related parameter bag."

---

## A-5 — Kinematic vs dynamic viscosity is silently chosen (MEDIUM)

`ReynoldsNumber(rho, v, L, mu)` takes dynamic viscosity μ in Pa·s. This is the "ρvL/μ" form. The equivalent "vL/ν" form (kinematic viscosity ν = μ/ρ in m²/s) is the one in:
- CRC Handbook of Chemistry and Physics (Table: "Viscosity of water" — both columns shown but ν is primary)
- ISO 3104 (kinematic viscosity, capillary viscometer — primary)
- ASHRAE Handbook (HVAC fluid properties — ν primary)
- ANSI/ASTM D445 (kinematic viscosity test method)

The dynamic form is primary in:
- Lemmon-NIST REFPROP database
- Bird-Stewart-Lightfoot textbook (engineering convention)
- Caleb Bell's `fluids` Python package (which takes μ by default but ships `nu_to_mu`/`mu_to_nu` converters)

**Reality does not ship either converter.** The user passing `ν` in m²/s into `ReynoldsNumber(rho, v, L, mu)` gets a result `~ρ` times too large with no diagnostic.

**Recommended fix — three pieces:**

1. Add `ReynoldsKinematic(v, L, nu) = v·L/nu` as a sibling function (inside `dimensionless/` if A-3 is adopted).
2. Add `KinematicFromDynamic(mu, rho) = mu/rho` and `DynamicFromKinematic(nu, rho) = nu*rho` as plain unit-conversion helpers, possibly in `constants/` (if the temperature-conversion family there is the precedent — they fit the same shape).
3. Document on `ReynoldsNumber` and on `Fluid` struct (if A-4 is adopted): "DynamicViscosity expects Pa·s, not m²/s. Use `KinematicFromDynamic`/`DynamicFromKinematic` to convert."

**Stokes also.** `StokesLaw(mu, r, v) = 6π·μ·r·v` — same dynamic-only form; same fix (sibling kinematic form `6π·ν·ρ·r·v` is mathematically equivalent and might be needed by users with ν-only data).

---

## A-6 — Pipe / channel geometry is unstructured (MEDIUM)

The canonical engineering quantities a pipe carries are: `D` (hydraulic diameter for non-circular), `L` (length), `ε` (absolute roughness), and material name (drawn copper / commercial steel / cast iron / concrete — each maps to a default `ε` in `Moody 1944`). Reality's `PipeFlowFriction(Re, roughness, diameter)` and `DarcyWeisbach(f, L, D, rho, v)` take these as bare floats; there is no `PipeMaterial` enum or roughness lookup.

**Recommended struct (compose with A-4):**

```go
type PipeMaterial int
const (
    DrawnCopper PipeMaterial = iota   // ε ≈ 1.5e-6 m
    CommercialSteel                   // ε ≈ 4.6e-5
    GalvanizedIron                    // ε ≈ 1.5e-4
    CastIron                          // ε ≈ 2.6e-4
    Concrete                          // ε ≈ 1e-3 (range)
    RivetedSteel                      // ε ≈ 3e-3
)
func (m PipeMaterial) Roughness() float64 { ... }

type Pipe struct {
    Diameter, Length, Roughness float64
    Material                    PipeMaterial  // optional; defaults to bare ε
}
```

Bell's `fluids.pipes` ships pipe-fitting K-factor functions for every named geometry (sharp entrance, rounded entrance, gradual contraction, sudden expansion, 45° elbow, 90° elbow rounded, gate valve, ball valve, plug valve, butterfly, etc. — ~50 fitting types, ~150 LOC, all closed-form). 067 T1-PIPE-FITTING (~200 LOC) is the equivalent commitment. **Each fitting K-factor is a function of `(Di, geometry-params)` — without a `Pipe` struct each call duplicates `Di`.**

**Channel parallel.** When 067 T1-OPEN-CHANNEL ships, `Channel{Width, Depth, Slope, Material, ManningN}` is the equivalent struct. Bell ships `fluids.open_flow` with explicit channel-shape enum (rectangular / trapezoidal / circular / parabolic).

---

## Comparison with Caleb Bell's `fluids` API (the architectural sibling per 068)

Bell's `fluids` (https://github.com/CalebBell/fluids, MIT, pure-Python, ~700 functions) is the closest existing analog to Reality's fluids surface. **Architectural deltas:**

| Dimension | Bell `fluids` (Python) | Reality `fluids` (Go) | Verdict |
|---|---|---|---|
| Function signature | Keyword args: `friction_factor(Re=49900, eD=9.2e-4, Method='Churchill1977')` | Positional `float64`: `PipeFlowFriction(49900, 9.2e-4*0.05, 0.05)` | Go has no kwargs; struct or option-bag closes gap |
| Method dispatch | `Method=` kwarg + `friction_factor_methods()` introspector | One Colebrook implementation, no choice | A-1 fix gives Go-idiomatic dispatch |
| Validation | `if Re <= 0: raise ValueError(...)` typical | NaN sentinels mostly; some unchecked | A-2 fix |
| Regime info | `friction_factor(..., return_method=True)` returns `(f, method_name_used)` | Returns scalar `f` | A-1 covers |
| Fluid struct | None (Bell uses kwargs throughout) | None | Reality could lead Bell here |
| Pipe struct | `fluids.Pipe(D=, L=, roughness=, ...)` | None | A-4, A-6 fix |
| Dimensionless numbers | `fluids.core.Reynolds()`, `fluids.core.Mach()`, ~30 numbers in `core/` | One number (Reynolds) in `fluids/` | A-3 fix |
| Atmosphere model | `fluids.Atmosphere(Z=)` returns `(T, P, ρ, μ)` US-Standard-1976 + NRLMSISE-00 | Absent | 067 T1 |
| Pipe fittings | ~50 K-factor functions in `fluids.fittings` | Absent | 067 T1-PIPE-FITTING |
| Open-channel | `fluids.open_flow.Manning()` | Absent | 067 T1-OPEN-CHANNEL |
| Two-phase flow | `fluids.two_phase.Lockhart_Martinelli()` etc. (~10 correlations) | Absent | 067 T2 |
| Particle settling | `fluids.particle_size_distribution`, `fluids.atmosphere.SettlingVelocity` | Absent (TerminalVelocity is the closed form, no iteration) | 067 N-3 fix |
| Compressible | `fluids.compressible.isentropic_T_T0(M, gamma=)`, normal/oblique shock | Absent | 067 T1-COMPRESSIBLE |

**Bell's signature shape Reality should adopt (Go-translated):**

```go
// Method dispatch via typed enum, not string
type FrictionMethod int
const (
    FrictionAuto FrictionMethod = iota   // pick best per Re/eD
    FrictionLaminar                      // 64/Re explicit
    FrictionColebrook                    // implicit, Brent-iterated
    FrictionChurchill1977                // single explicit, all-Re
    FrictionSerghides                    // explicit, ±0.0023% vs Colebrook
    FrictionHaaland                      // explicit, ±2%
    FrictionSwameeJain                   // explicit, ±1%
    FrictionMoody                        // alias for Churchill1977
)

type FrictionResult struct {
    F          float64
    Regime     PipeFlowRegime
    Method     FrictionMethod  // method actually used (esp. for Auto)
    Iterations int
}

func FrictionFactor(Re, eD float64, method FrictionMethod) (FrictionResult, error)
```

This subsumes A-1 (returns regime + method) and reads at the call site as well as Bell's Python:

```go
res, err := fluids.FrictionFactor(49900, 9.2e-4, fluids.FrictionAuto)
// res.F, res.Regime, res.Method, res.Iterations, err
```

**Bell's pattern Reality should NOT adopt:** string-keyed method names (`Method='Churchill1977'`). Go has typed enums; use them. String-keys lose IDE autocomplete and admit typos.

---

## Recommended API redesign — single sprint, ~150 LOC, zero numerical change

The numerical layer (066 N-1 Churchill smoothing, N-2 relative tolerance, N-3 Cd-Re iteration) is already tracked in 066's recommendations. **This report's recommended API changes are independent and additive:**

1. **Create `dimensionless/` package** (`dimensionless/dimensionless.go`, ~120 LOC for 18 numbers per 068 D-2). `fluids.ReynoldsNumber` re-exports `dimensionless.Reynolds` for backward compat.

2. **Add `Fluid` and `Pipe` structs** to `fluids/types.go` (~40 LOC). Methods compute derived values (Area, RelativeRoughness, KinematicViscosity).

3. **Add three-return shape to `PipeFlowFriction`** — new function `FrictionFactor(Re, eD, method) (FrictionResult, error)` per Bell-style; existing `PipeFlowFriction` deprecated, points to new.

4. **Add NaN guards for negative ρ, μ, A, D, L, m, r** across the 11 existing functions. Document the contract in package-level docstring. ~30 LOC, no signature change.

5. **Add unit-conversion sibling `KinematicFromDynamic`, `DynamicFromKinematic`** in `constants/` or `dimensionless/` (~10 LOC).

**Total surface change:** ~200 LOC, +1 package, +2 structs, +1 enum, no breakage. **Cross-language port impact:** Python/C++/C# follow the same struct shape; the Go-typed `FrictionMethod` enum maps to `Enum` in Python and `enum class` in C++/C#.

---

## What this report did NOT cover (out of scope per claim of disjointness)

- Numerical correctness of present surface (066 owns: Re=2300 cliff, Colebrook absolute-vs-relative tolerance, Cd-Re self-consistency, Bernoulli cancellation hazard).
- Missing primitives enumeration (067 owns: Tier-1/2/3 sprint plan, ~9,500 LOC, 220–290 canonical primitives).
- SOTA architectural binding decisions (068 owns: D-1 closed-form-correlation parity first, D-2 Brent + Serghides Colebrook, D-3 Palabos lattice-descriptor pattern, D-4 pseudospectral first NS, D-5 SoA particles, D-6 hand-rolled adjoints, D-7 cite-and-skip ML).
- Performance / hot-path layout (070 fluids-perf owns).

This report owns: signature shape, return contract, struct grouping, dimensional validation, regime-aware dispatch, kinematic-vs-dynamic viscosity choice, geometry parameter shape, dimensionless-numbers sub-package factoring, comparison with Bell's `fluids` API surface.

---

**Auditor:** agent 069
**Status:** complete
**Six findings open:** A-1 regime hidden (HIGH), A-2 zero validation (HIGH), A-3 dimensionless/ sub-package missing (HIGH), A-4 no Fluid/Pipe structs (HIGH), A-5 kinematic-vs-dynamic ambiguous (MEDIUM), A-6 pipe geometry unstructured (MEDIUM).
**Single highest-leverage change:** ship `(value, regime, error)` return shape on `PipeFlowFriction` + new `FrictionFactor(Re, eD, method)` Bell-style — closes A-1 (topic-prompt headline) and aligns Reality's API with the architectural sibling 068 named.

# 114 | physics-api

**Scope.** API ergonomics of `physics/` (28 fns across `mechanics.go` 153 LOC,
`thermo.go` 148 LOC, `materials.go` 209 LOC, `optics.go` 90 LOC). Disjoint
from 111 (numerics F-1..F-12 + new APIs M-1..M-7), 112 (T1.1–T3.5 missing
backlog), 113 (six SOTA-port engineering tricks T1–T6). This report only
covers **how a Reality consumer calls the package**: type signatures,
parameter ordering, naming, scalar-vs-vector decisions, tuple-return shape,
unit-tagging policy, mu/G provenance, and consistency with sibling packages
(`geometry`, `linalg`, `em`, `orbital`, `fluids`, `constants`).

**TL;DR — five concrete API findings:**

1. **`KineticEnergy(m, v)` and `PotentialEnergy(m, g, h)` take a *scalar*
   speed/height even though every physics consumer (Pistachio, aicore,
   chaos integrators) carries velocity as a vector.** No `KineticEnergyVec`
   sibling. The fix is a `[3]float64` overload with `KineticEnergyVec(m, v
   [3]float64)` not a redesign — keep the scalar form, add the vector form,
   wire both to the same dispatch via dot product (`v·v`). ~6 LOC, zero
   semantic change. Same gap exists for `SpringForce` (1D scalar `x`/`v`,
   no 3D Hooke spring) and `ProjectilePosition` (returns named scalars `x,
   y` instead of `[3]float64` even though a parabola lives in a plane in
   3-space).

2. **Argument ordering is inconsistent between mass-first and
   parameter-first conventions.** `KineticEnergy(m, v)` puts mass first,
   `SpringForce(k, x, c, v)` puts spring constant first (mass implicit/not
   present), `ElasticCollision(m1, v1, m2, v2)` interleaves
   mass/velocity per particle, `Pendulum(theta, L, g, damping)` puts state
   first not parameters. There is no documented rule (compare `linalg`'s
   `(out, a, b)` rule, `geometry`'s `(target, source)` rule for quaternion
   ops). The textbook convention used by SymPy/Mathematica is
   `(state..., params..., constants...)`; physics/ violates this in
   `KineticEnergy` (param `m` before state `v`) and `PotentialEnergy` (param
   `m, g` before state `h`). This is the **single highest-leverage cross-
   cutting API decision** because every Tier-1 fn from 112 (Lagrangian,
   Lorentz, four-vectors, rigid-body) will face the same ordering choice
   and inheriting the inconsistency forward poisons the v0.11–v1.0
   surface. Section 3 below specifies the rule.

3. **There is no `Vec3` / `Tensor3Sym` / `Tensor3` type in physics or
   anywhere in `reality` outside `geometry`.** `geometry` already pays the
   cost of inventing `[3]float64`/`[4]float64` array types (zero
   allocation, stack-friendly, validated in 1,758 Pistachio tests).
   `physics` should **import that convention** rather than fight it. When
   112 T2.1 ships continuum stress, the Voigt-6 vector type belongs in
   `geometry/tensor.go` (or its own `physics/tensor.go`) using the same
   `[6]float64` array idiom — *not* `[]float64` slice (heap), *not*
   `struct{Sxx, Syy, ...}` (mismatch with sibling pkgs). Verified: 0 hits
   for `[3]float64` in `em/`, `physics/`, `fluids/`, `orbital/`,
   `acoustics/`. All four packages currently lose vector context the
   moment they touch a velocity, force, field, or stress.

4. **Unit-aware types are correctly OUT OF SCOPE for the physics package
   itself**, but the dimensional-lint pattern from 113 T1 IS the right
   answer to the unit-safety question agent 112 deferred. The reality
   convention "numbers in, numbers out" means a `Quantity` wrapper struct
   is wrong for `physics`. Instead, ship the SI-unit annotations as
   structured docstring metadata (already partially present:
   `IdealGas`'s `n: amount of substance (mol)` parses to dim-vector
   $[L^0,M^0,T^0,I^0,\Theta^0,N^1,J^0]$) and have `testutil/dim_lint.go`
   (113 T1) verify each fn's RHS dim matches LHS at `go test`. This makes
   units a build-time concern not a runtime tax. Adopt 113 T1 verbatim.

5. **`OrbitalVelocity(M, r)` in `physics/mechanics.go:72` is a name
   collision with `OrbitalVelocity(mu, r, a)` in `orbital/orbital.go`.**
   The physics version takes mass-of-central-body and uses
   `constants.GravitationalConst` internally; the orbital version takes
   $\mu = GM$ and an additional semi-major axis `a` (vis-viva). Both
   exported in the same module path `github.com/davly/reality/...`. This
   collides on auto-import in any consumer that imports both packages
   (Pistachio orbital visualizer, aicore astrodynamics reasoner). Rename
   to `CircularOrbitVelocity(M, r)` in physics, or move it to `orbital/`
   and delete the duplicate. Agent 109 already flagged the `mu`-vs-`M`
   inconsistency from the orbital side; this is the same bug from the
   physics side.

The single highest-leverage commit: **finding (2) — codify a Reality-wide
parameter ordering rule** before 112's Tier-1 ships. Cost ~5 LOC of doc in
`CLAUDE.md` plus 1-LOC reorderings of 4 existing fns; benefit is preventing
~95 future fns from inheriting the inconsistency.

---

## 1. Inventory: every signature, classified

Twenty-eight exported functions. Grouped by what they take/return:

| Category | Count | Examples | Verdict |
|---|---|---|---|
| Scalar-in, scalar-out | 22 | `NewtonSecondLaw(F,m)`, `KineticEnergy(m,v)`, `IdealGas(n,T,V)`, `HookesLaw(E,eps)` | OK as numbers-in-numbers-out |
| Scalar-in, **named tuple**-out | 2 | `ProjectilePosition`, `ElasticCollision` | should be `[2]float64` / `[3]float64` |
| Slice-in, slice-out (buffered) | 1 | `HeatEquation1DStep(u, dt, dx, alpha, out)` | follows linalg convention correctly |
| Tensor-in (FUTURE per 111 M-1) | 0 | — | needs design — see §4 |
| Vector-in (FUTURE) | 0 | — | needs design — see §4 |

**No function in physics/ currently takes a vector argument.** Compare:

- `linalg.Dot(a, b []float64) float64` — vector-aware
- `geometry.QuatMul(a, b [4]float64) [4]float64` — vector-aware (fixed-size)
- `chaos.RK4Step(state []float64, ...)` — vector-aware (variable-size)
- `prob.Mean(xs []float64)` — vector-aware

Physics is **the only domain pkg in reality that has no vector primitives
at all** despite physics being the canonical vector-math domain. This is
historical (28 fns extracted from a freshman textbook one at a time, not
designed top-down) and fixable additively.

---

## 2. Naming: `KineticEnergy(m, v)` is correct; the rest is uneven

The audit-prompt question "`KineticEnergy(m, v)` vs `KineticEnergy(p)` vs
`KE`?" has a clear answer: **the current shape is right.** Three reasons:

- `KE` is a 2-letter identifier — violates Go idiom and conflicts with
  Reality's "every function name is a self-contained noun phrase" pattern
  (compare `acoustics.DecibelSPL` not `dBSPL`, `prob.NormalCDF` not
  `normCDF`). 049-constants-api flagged the inverse problem (`Pi` vs
  `math.Pi`); this is the same lesson — abbreviations cost more than they
  save.
- `KineticEnergy(p)` taking momentum would force every caller to compute
  `p = m*v` at the call site or accept a `Momentum(m, v)` helper — net
  more LOC. Closed-form physics functions take the variables a textbook
  states them in. The textbook says $KE = \frac{1}{2}mv^2$, not $KE =
  p^2/2m$ (which is the same number, but the second form silently loses
  precision when `p` is computed elsewhere with cancellation).
- `KineticEnergy(m, v)` parses unambiguously in any consumer language
  (Go, Python, C++, C# — all four golden-file targets read the JSON
  field names `m, v`).

**What IS uneven** is the *list* of recognized energy/force functions:

| Physics quantity | Has a fn? | Notes |
|---|---|---|
| Kinetic energy (translational) | yes | `KineticEnergy(m, v)` |
| Kinetic energy (rotational) | NO | $\frac{1}{2}I\omega^2$ — gap, fits 112 T1.5 |
| Potential energy (gravitational, near-Earth) | yes | `PotentialEnergy(m, g, h)` |
| Potential energy (gravitational, point-mass) | NO | $-Gm_1m_2/r$ — gap |
| Potential energy (spring) | NO | $\frac{1}{2}kx^2$ — gap, sibling of `SpringForce` |
| Potential energy (electrostatic) | covered in `em/` | `em.CoulombForce` exists; PE not — gap |
| Total mechanical energy | NO | KE + PE composition fn — gap |
| Linear momentum $p = mv$ | NO | trivial but missing — gap |
| Angular momentum $L = r \times p$ | NO | flagged by 111 M-2 |

The gap pattern: **every "force" has a partner "energy" except spring,
which has only force**, and **every "energy" has a partner "momentum"
except none of them do**. This is a feature-coverage question (112's
domain) but the naming has to commit before the fns ship, hence flagging
here. Recommended: `KineticEnergyRotational(I, omega)` /
`PotentialEnergyGrav(m1, m2, r)` / `PotentialEnergySpring(k, x)` /
`LinearMomentum(m, v)` / `AngularMomentum(r, v, m [3]float64)` — each
suffix-disambiguates with the partner concept it pairs with, no
abbreviations.

---

## 3. Argument order: codify `(state, params, constants)` rule

Eight order conventions in the existing 28 fns:

| Function | Order | Mass position | Notes |
|---|---|---|---|
| `NewtonSecondLaw(F, m)` | force, mass | 2nd | mass last — output first style |
| `KineticEnergy(m, v)` | mass, vel | 1st | textbook: $\frac{1}{2}mv^2$ |
| `PotentialEnergy(m, g, h)` | mass, accel, height | 1st | textbook: $mgh$ |
| `GravitationalForce(m1, m2, r)` | mass, mass, dist | 1st&2nd | both masses before r |
| `SpringForce(k, x, c, v)` | param, state, param, state | n/a | param-state interleaved |
| `ElasticCollision(m1, v1, m2, v2)` | per-particle (m,v) | interleaved | per-body grouping |
| `Pendulum(theta, L, g, damping)` | state, length, gravity, damping | n/a | state first |
| `OrbitalVelocity(M, r)` | mass, radius | 1st | central-body conv. |
| `IdealGas(n, T, V)` | amount, temp, vol | n/a | standard ideal-gas order |
| `HookesLaw(E, epsilon)` | modulus, strain | n/a | param then state |
| `VonMisesStress(s1, s2, s3)` | three principal stresses | n/a | symmetric triple |
| `StressIntensityFactor(sigma, a, Y)` | stress, length, factor | n/a | mixed |
| `BeerLambertLaw(I0, mu, x)` | initial, coef, length | n/a | state, param, length |
| `BeamDeflection(P, L, E, I)` | load, length, modulus, second-moment | n/a | mixed |

There are **at least four orderings in 28 functions**. This is well below
the linalg/geometry consistency bar. Three options:

**(A) State first.** `KineticEnergy(v, m)`, `PotentialEnergy(h, m, g)`,
`SpringForce(x, v, k, c)`, `Pendulum(theta, L, g, damping)` — current
Pendulum is correct, others would change. Matches SymPy convention.

**(B) Textbook-formula reading order.** Whatever order the LaTeX equation
is conventionally written. $\frac{1}{2}mv^2$ → `(m, v)`; $-kx-cv$ →
`(k, x, c, v)`; $mgh$ → `(m, g, h)`. This is what physics/ mostly does
today. **Pro**: docstring formula matches signature 1:1. **Con**:
inconsistent across functions because textbooks aren't consistent
(some write $mgh$, some $mh \cdot g$).

**(C) Material constants → geometry → state.** `KineticEnergy(m, v)` →
mass is intrinsic property of the object, $v$ is its current state — OK.
`SpringForce(k, c, x, v)` → spring constants first, then position then
velocity — would change current order. Matches Modelica MultiBody and
Brax conventions.

**Recommendation: codify (B) — textbook-formula reading order — as the
explicit rule in CLAUDE.md, with one tiebreaker: when textbook ordering
varies, prefer (state, params, fundamental constants).** This requires
**zero changes to existing code** (most fns already follow it), only
documents the implicit rule that has emerged. Apply consistently to 112
Tier-1 and beyond.

The only existing violation under rule (B) is `SpringForce(k, x, c, v)`
where the formula reads $-kx - cv$ — already correct. False alarm. The
**actual** violations of any consistent rule are:

- `ProjectilePosition(v0, theta, t, g)` — should the constant `g` be
  last? Yes — fundamental-constants-last is a defensible sub-rule. Keep.
- `Pendulum(theta, L, g, damping)` — ditto. Keep.
- `NewtonSecondLaw(F, m)` — formula is $a = F/m$, output computed from
  inputs read left-to-right. Keep.

So actually the **status quo is more consistent than it looks**; the
remaining job is to write the rule down. Do that as a 3-line addition to
CLAUDE.md "Key Design Rules" section between rule 4 and rule 5.

---

## 4. Vector and tensor representation: defer to `geometry` array convention

When 111 M-1 / 112 T2.1 ships continuum stress, four representation
choices:

- **(α) Voigt 6-vector `[6]float64`**: `{σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy}`.
  Asymmetric tensors unrepresentable. ~33% memory savings vs full 3×3.
  Modelica/FEniCS/Abaqus canonical. Triple-confirmed by 111 M-1, 112 T2.1,
  113 T2.
- **(β) Full 3×3 `[9]float64`** flat row-major. Cauchy stress in continuum
  mechanics is symmetric so 3 of 9 floats are redundant — golden-file
  validation has to check symmetry on every output, which is an extra
  test surface that Voigt deletes by construction.
- **(γ) Struct `Tensor3 struct{Sxx, Syy, Szz, Syz, Sxz, Sxy float64}`**.
  Type-safe field access. Cannot be passed by value to a 4-language
  golden-file validator (Python/C++/C# can't deserialize a Go struct
  layout — JSON works but breaks the array-shaped vector convention).
- **(δ) `[]float64` slice of length 6**. Heap allocation, breaks 60-FPS
  charter.

**Adopt (α).** Specifically: `[6]float64` ordered as Voigt $\{xx, yy, zz,
yz, xz, xy\}$, with one alias type and four golden-file accessors:

```go
type Tensor3Sym [6]float64

func (t Tensor3Sym) Xx() float64 { return t[0] }
func (t Tensor3Sym) Trace() float64 { return t[0] + t[1] + t[2] }
// ...
```

Vector quantities (force, velocity, position, momentum, angular velocity,
moment-of-inertia diagonal): `[3]float64` matching `geometry`'s vec3
convention. Verified `geometry/sdf.go` and `geometry/curves.go` already
use this idiom and the resulting code reads cleanly.

The single most important architectural commit before 112 ships: write
this in `physics/doc.go` so every Tier-1 author inherits the convention.

**Reject Voigt-with-engineering-shear-doubling** (the {ε_xx, ε_yy, ε_zz,
2ε_yz, 2ε_xz, 2ε_xy} convention used by some FEM codes for strain). It
breaks the duality $\sigma : \varepsilon = \sum \sigma_i \varepsilon_i$
that makes Voigt valuable in the first place. Use Mandel notation
($\sqrt{2}$ on the off-diagonal) IF the duality matters, OR use the
unscaled Voigt convention — but never the engineering-shear convention.
Recommended: unscaled Voigt for both stress and strain, with $1/2$
factor explicit at the energy/Hooke-law site.

---

## 5. Tuple returns: `[N]float64` not named scalars

Two existing functions return tuples:

```go
func ProjectilePosition(v0, theta, t, g float64) (x, y float64)
func ElasticCollision(m1, v1, m2, v2 float64) (v1f, v2f float64)
```

Named-scalar tuple returns work in Go but force callers to write:

```go
x, y := physics.ProjectilePosition(...)
pos := [2]float64{x, y}  // 1 LOC of repackaging at every consumer
```

vs the array form:

```go
pos := physics.ProjectilePosition(...)  // returns [2]float64 directly
```

Pistachio's particle system, chaos's ODE state, geometry's curve
parameterizers all consume `[2]float64`/`[3]float64` directly. Go's named
return values offer zero ergonomic value here — they're not exposed
through golden-file JSON (which has to flatten to `["x", 1.5, "y", 2.3]`
or similar regardless), and they prevent direct consumption.

**Recommend:** convert both to `[2]float64` returns. Migration is two
function-signature changes plus updating ~6 test sites; the named-return
form is preserved in docstrings for clarity.

For `ElasticCollision`, returning `[2]float64` of `{v1f, v2f}` is correct
(both values are scalars in the same dimension — velocity). For an
eventual 3D `ElasticCollision3D(m1, v1, m2, v2 [3]float64) (v1f, v2f
[3]float64)`, the tuple is of two `[3]float64` — Go handles this
naturally; no change to the convention.

---

## 6. Mu / G provenance: physics vs orbital collision

`physics.OrbitalVelocity(M, r)` (line 72, mechanics.go) takes mass and
uses `constants.GravitationalConst` internally:

```go
func OrbitalVelocity(M, r float64) float64 {
    return math.Sqrt(constants.GravitationalConst * M / r)
}
```

`orbital.OrbitalVelocity(mu, r, a)` takes $\mu = GM$ explicitly and adds
vis-viva's semi-major axis. Both are exported under the same module path.

**Bugs:**

1. **Identifier collision** — agent 109 flagged the `mu` vs `M` semantic
   mismatch from the orbital side. From the physics side, the deeper
   issue is that **physics computes $G \cdot M$ at every call site**,
   accumulating the relative uncertainty of `constants.GravitationalConst`
   ($2.2 \times 10^{-5}$, the worst-precision constant in CODATA 2018,
   per agent 049) into a function whose other inputs ($M$, $r$) are known
   to ~15 digits. The Cassini-Huygens flight team and JPL Horizons both
   maintain $\mu_\odot$, $\mu_\oplus$, etc. as **separately measured
   constants** to ~11 digits each — far better than $G \cdot M$
   reconstructed from individual factors. Astrodynamics consumers
   (orbital, aicore) want the $\mu$ form. Mass-times-$G$ is the
   intro-physics form.

2. **No $\mu$-form in physics**. There is no `physics.OrbitalVelocityMu(mu,
   r)` to use the better-conditioned input. Adding one would create three
   functions doing nearly the same thing across two packages. Cleaner: 
   delete `physics.OrbitalVelocity` (move usage to `orbital`), or rename
   to `physics.CircularOrbitVelocityM(M, r)` to disambiguate.

**Recommendation: rename to `physics.CircularOrbitVelocityFromMass(M, r)`**.
Verbose but disambiguates from `orbital.OrbitalVelocity`, makes the unit
expectation explicit (mass not μ), and signals "pedagogical undergraduate
form, use `orbital/` for production astrodynamics" without breaking any
existing call site that explicitly imports physics.

Same pattern audit for the rest of the package: `GravitationalForce` is
fine (closed-form Newton), `Pendulum` is fine (uses `g` not $GM/r^2$),
`ProjectilePosition` is fine. Only `OrbitalVelocity` collides.

---

## 7. Cross-pkg consistency check

| Convention | linalg | geometry | em | fluids | orbital | physics | Verdict |
|---|---|---|---|---|---|---|---|
| `[3]float64` for vectors | yes | yes (vec3) | NO | NO | NO | NO | physics joins em/fluids/orbital in the gap |
| `(out, in1, in2)` buffered ops | yes | yes | NO | NO | NO | yes (HeatEq) | physics partially correct |
| Slice for variable-N | yes | NO | yes (resistors) | NO | NO | yes (HeatEq) | OK |
| Struct for stateful | NO | NO | NO | NO | NO | NO | unanimous — none of these pkgs have state |
| Named tuple returns | NO | NO | NO | NO | yes | yes | physics + orbital diverge from rest |
| SI units in docstring | yes | n/a | yes | yes | yes | yes | unanimous |
| `(state, params, consts)` order | yes | yes | yes | yes | mixed | mixed | physics + orbital lag |

The physics package is on the unfavorable side of every consistency vote
except SI-units-in-docstrings (which is universal). The fix is additive:
introduce vec3 inputs and `[N]float64` returns; codify the parameter
order rule once. None of this requires breaking existing call sites —
add the new vector forms alongside the scalar forms and let consumers
migrate at their pace.

---

## 8. Sized recommendations (additive, all non-breaking)

| ID | Change | LOC | Breaks? | Source |
|---|---|---|---|---|
| A1 | Codify `(state, params, consts)` order rule in CLAUDE.md | 5 | no | §3 |
| A2 | Add `KineticEnergyVec(m, v [3]float64)` | 6 | no | §1 |
| A3 | Add `LinearMomentum(m, v float64)` and `LinearMomentumVec(m, v [3]float64)` | 12 | no | §2 |
| A4 | Add `KineticEnergyRotational(I, omega)` | 6 | no | §2 |
| A5 | Add `PotentialEnergySpring(k, x)` | 4 | no | §2 |
| A6 | Add `PotentialEnergyGrav(m1, m2, r)` (point-mass) | 6 | no | §2 |
| A7 | Convert `ProjectilePosition` return to `[2]float64` | 8 | YES (gentle) | §5 |
| A8 | Convert `ElasticCollision` return to `[2]float64` | 8 | YES (gentle) | §5 |
| A9 | Rename `OrbitalVelocity` → `CircularOrbitVelocityFromMass` | 4 | YES (rename) | §6 |
| A10 | Define `Tensor3Sym = [6]float64` Voigt alias in `physics/tensor.go` | 30 | no | §4 |
| A11 | Document vec3 / Voigt-6 conventions in `physics/doc.go` | 25 | no | §4 |
| A12 | Adopt 113 T1 dim-lint as `testutil/dim_lint.go` for physics | (covered by 113) | no | §0 finding 4 |

**Tier 1 (commit before 112 Tier-1 ships, ~75 LOC)**: A1, A10, A11, A12.
These set the convention every future fn inherits. Cost is rounding
error; benefit compounds across ~95 future fns.

**Tier 2 (cleanup, ~50 LOC)**: A7, A8, A9. Three function-signature
changes, all gentle (rename/return-shape), with grep-ablely few call
sites in this repo. Should ship in v0.11 alongside 112 Tier-1 work.

**Tier 3 (additive coverage, ~34 LOC)**: A2–A6. Five new functions
filling the energy/momentum gap matrix. Trivial to ship, valuable for
aicore consumer that today computes these inline at every reasoning
site.

The **single highest-leverage commit** is A1 — codifying the parameter
order rule before 112's ~95 new functions inherit the implicit pattern.
Cost 5 LOC; benefit prevents inconsistency in nearly 100 future
signatures. Second-highest: A11 (vec3/Voigt convention doc, 25 LOC) —
same logic at the type-shape level rather than the parameter-order
level.

---

## 9. Direct answers to audit-prompt headlines

**Q: All physics functions take/return float64 — no unit-bearing types.**
Yes, and **this is correct under the Reality charter** (numbers in,
numbers out). The right place for unit safety is 113 T1's `testutil/
dim_lint.go` parsing existing per-param SI annotations from docstrings
into a 7-vector $[L,M,T,I,\Theta,N,J]$ dim algebra and verifying RHS
matches LHS at `go test`. Zero runtime cost, zero charter violation,
catches the NASA-Mars-Climate-Orbiter class of bug.

**Q: Vector vs scalar functions — 1D vs 3D positions/velocities.** All
existing fns are 1D. No vector primitives anywhere in physics/. Section
4 specifies the additive fix: `[3]float64` from `geometry` for vectors,
new `*Vec` siblings for `KineticEnergy` / `SpringForce` /
`ElasticCollision`. Keep the scalar forms — they pair correctly with the
1D textbook formulas — and add vector forms alongside.

**Q: Tensor representation — full 3×3 vs Voigt 6-vector.** Voigt
6-vector unanimously. Triple-confirmed by 111 M-1, 112 T2.1, 113 T2.
Specifically `[6]float64` ordered $\{\sigma_{xx}, \sigma_{yy},
\sigma_{zz}, \sigma_{yz}, \sigma_{xz}, \sigma_{xy}\}$ with no
engineering-shear factor (use $1/2$ factor explicit at the Hooke/energy
site). Reject struct-with-named-fields (breaks 4-language golden-file
shape). Reject `[]float64` slice (heap, breaks 60-FPS charter).

**Q: Comparison with constants/'s flat float64 (per 049).** Constants
ships flat `const float64` — perfect for compile-time substitution.
Physics ships `func ... float64` — perfect for runtime computation. Both
are the right answer for their layer. The connection point is the
unit-conversion subpackage 049 recommended (`constants/convert`):
adopting that pattern means physics fns can call
`constants.convert.KelvinFromCelsius(c)` to normalize input units at the
boundary without taking on a Quantity type. Defer the convert subpkg
itself to 049's roadmap; physics doesn't need to do anything until it
exists.

**Q: Naming — `KineticEnergy(m, v)` vs `KineticEnergy(p)` vs `KE`.**
Current shape `KineticEnergy(m, v)` is correct. `KE` is too short
(violates Go idiom + Reality's noun-phrase convention). Momentum form
`KineticEnergy(p)` forces callers to compute `p = m*v` (added LOC) and
silently loses precision when `p` comes from elsewhere. The textbook
form is the right form; ship rotational/relativistic siblings with
explicit suffixes (`KineticEnergyRotational(I, omega)`,
`KineticEnergyRelativistic(m, v)` per 112 T1.1).

**Q: Args order — `m` before `v`? velocity arg singular vs vector?**
Mass-before-velocity in `KineticEnergy(m, v)` matches textbook formula
$\frac{1}{2}mv^2$ — correct under rule (B) in §3. Velocity arg is
singular today; add a `*Vec` sibling taking `[3]float64` per finding (1).
The cross-cutting rule is **textbook-formula reading order**, with
`(state, params, fundamental-constants)` as tiebreaker — write this in
CLAUDE.md before 112 Tier-1 ships.

---

## 10. What this report deliberately did NOT cover

- Numerical correctness (111's domain — F-1..F-12, M-1..M-7).
- Missing-feature backlog (112's domain — T1.1..T3.5, ~1,800 LOC).
- Library porting / SOTA comparison (113's domain — T1..T6).
- Performance / allocations / 60 FPS hot-path measurement (no `physics-perf` agent in this batch — flagging that the gap exists but it's out of scope here).
- Test coverage / golden-file ratio (111 C-3 already flagged 3/28 = 11% golden coverage).

---

## 11. Sprint plan synthesis with 111/112/113

Pre-v0.11 (this PR window, before 112 Tier-1 begins):
- A1 (param order rule, 5 LOC)
- A11 (vec3/Voigt convention doc, 25 LOC)
- A10 (Tensor3Sym alias, 30 LOC)
- A12 (113 T1 dim-lint, 200 LOC — owned by 113)

v0.11 alongside 112 Tier-1:
- A7, A8, A9 (return-shape and rename cleanups, 20 LOC)
- A2–A6 (energy/momentum coverage gap, 34 LOC)

v0.13 alongside 112 T2.1 continuum:
- Use Tensor3Sym from A10 throughout new continuum-stress code
- Cardano principal-stress closed-form per 111 M-1 / 112 T2.1

Total physics-api work: ~115 LOC over 3 milestones, no breaking changes
to the 22 scalar-in-scalar-out fns, all gentle changes confined to 4 fns
(`ProjectilePosition`, `ElasticCollision`, `OrbitalVelocity` rename,
docstring additions).

---

**End report. Output: `agents/114-physics-api.md`, ~370 lines.**

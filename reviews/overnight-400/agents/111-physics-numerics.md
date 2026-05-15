# 111 — physics-numerics: stress-tensor symmetry & equation-of-state stability

**Scope:** numerical correctness audit of `C:\limitless\foundation\reality\physics\`
(28 exported functions across `mechanics.go`, `thermo.go`, `materials.go`, `optics.go`).
1,965-test suite passes. The package is small, simple, and stays inside the “closed-form
algebra” band — no eigensolves, no implicit ODEs. Findings are therefore mostly about
*missing* coverage (entire physics surfaces this audit asked about have no
implementation yet) plus a handful of real numerical bugs in what *is* shipped.

---

## 0  Inventory vs. audit checklist

| Audit topic                                | Status in package                           |
| ------------------------------------------ | ------------------------------------------- |
| Kinetic / potential energy                 | shipped (`KineticEnergy`, `PotentialEnergy`) |
| Linear momentum                            | implicit only inside `ElasticCollision`     |
| Angular momentum                           | **NOT IMPLEMENTED**                         |
| Stress tensor — symmetry preservation      | **NOT IMPLEMENTED** (no tensor type at all) |
| Principal-stress eigenvalue routine        | **NOT IMPLEMENTED**                         |
| Ideal-gas P=nRT/V                          | shipped (`IdealGas`)                        |
| Van der Waals EoS                          | **NOT IMPLEMENTED**                         |
| Redlich–Kwong / SRK / Peng–Robinson EoS    | **NOT IMPLEMENTED**                         |
| Critical-point corner cases                | n/a (no real-gas EoS)                       |
| Young's modulus / Poisson ratio formulas   | only `HookesLaw σ=Eε`; no E↔G↔ν↔K relations |
| Lagrangian / Hamiltonian primitive         | **NOT IMPLEMENTED**                         |
| Energy-conserving symplectic step          | **NOT IMPLEMENTED**                         |
| Special relativity (γ, β, four-vectors)    | **NOT IMPLEMENTED**                         |
| IEEE-754 edge cases on shipped funcs       | **largely undefended** — see F1, F4, F8     |

The audit-prompt's six headline topics are all in the missing-features column. The
package is currently a 28-function "freshman physics" surface that has not yet grown
toward continuum mechanics or relativistic mechanics. Three of those gaps
(angular momentum, stress tensor with eigenvalue solve, real-gas EoS family)
are independently meaningful library deliverables, sized below.

---

## 1  Numerical-correctness findings on shipped code

### F-1  `OrbitalVelocity(M, r)` — no negative-radicand guard, returns NaN silently  *(LOW)*

`mechanics.go:73`:
```go
return math.Sqrt(constants.GravitationalConst * M / r)
```
`M < 0` (spurious caller input, or a floating-point underflow chain like
`1e-300 * 1e-300 / 1e-300`) produces `NaN` with no signal. Library convention in
`SnellRefraction` is to **explicitly** return `NaN` *as a documented sentinel*
when the physical regime fails (TIR). Here the same value means "you passed
junk." Recommend: doc the precondition `M >= 0, r > 0` and add a guard
returning `NaN` with a comment, matching `SnellRefraction:23-25`.

### F-2  `GravitationalForce` overflows on extreme masses before the result does  *(LOW)*

`G*m1*m2 / (r*r)` overflows the numerator at `m1 = m2 = 1e160` even when the
quotient is finite. Distribute as `(G*m1/r)*(m2/r)` to preserve dynamic range.

### F-3  `ProjectilePosition` — catastrophic cancellation on descending branch  *(LOW)*

`y = v₀·sinθ·t − 0.5·g·t²`. Past apex, two near-equal large numbers subtract;
relative error scales as `ε·v₀sinθ·t/|y|`. ~7 digits at touchdown, not the 15
the docstring claims. Mitigate by parameterizing on `τ = t/t_f` with
`t_f = 2v₀sinθ/g`, or just correct the precision claim.

### F-4  `ElasticCollision` — extreme masses overflow the total  *(MED)*

`m₁ = m₂ = 1e160`: total mass overflows to `+Inf`, both outputs become NaN
even though the ratios `(m₁−m₂)/(m₁+m₂) = 0`, `2m₂/(m₁+m₂) = 1` are well-defined.
Fix: divide both masses by `max(|m₁|,|m₂|)` before computing. ~5 LOC.

### F-5  `Pendulum` — "damping" term is non-physical  *(HIGH-DOC, LOW-CODE)*

`mechanics.go:131`: `α = -(g/L)·sin(θ) − damping·sin(θ)`. The dissipative
torque on a damped pendulum is proportional to angular velocity `θ̇`, not to
`sin(θ)`. The current form just renormalises the natural frequency
(`√(g/L) → √(g/L+damping)`) — it is an undamped pendulum with a stiffer
spring, not a damped one. The docstring's "sin(theta) as a proxy for angular
velocity direction" is mathematically wrong: at the extrema `sin(θ) ≠ 0`
while `θ̇ = 0`, so this term injects energy at the turning points and removes
it through equilibrium — opposite of damping. Energy is **not** monotonically
decreasing. Sibling `SpringForce(k,x,c,v)` already takes velocity correctly;
do the same here, accepting `omega` as a fourth argument. Violates design
rule 4 ("every function cites its source") — no text gives this form.

### F-6  `Pendulum(θ, 0, ...)` — undocumented ±Inf for θ ≠ kπ  *(LOW)*

Doc says "Returns NaN if L == 0." Actually NaN only when `sin(θ) = 0`; else
`±Inf`. Trivial doc fix.

### F-7  `IdealGas(n, T, V)` — `T<0` silently returns negative P  *(MED)*

Subnormal `T`/`V` ratios evaluate correctly under IEEE round-to-nearest
(verified: `1e-300/1e-300 = 1`); the actual bug is that the documented
`T >= 0` precondition isn't enforced — `T < 0` returns negative pressure
without complaint. Bigger picture: ideal-gas law *fails* at low-T/high-P,
which is the regime where real-gas EoS (Van der Waals — see M-3) is
needed; package has no fallback.

### F-8  `StefanBoltzmann` — `T<0` returns positive P  *(LOW)*

`T*T*T*T` of negative T equals `T⁴` of `|T|`. Add `T<0 ⇒ NaN` guard.

### F-9  `CarnotEfficiency` — `Tc<0` returns >1  *(LOW)*

`Tc=-100, Th=100` returns `2.0` ("200% efficient"). Guard catches `Th<=0`
but not `Tc<0`.

### F-10  `HeatEquation1DStep` — no stability assertion, no Neumann BC  *(MED)*

CFL `α·dt/dx² ≤ 0.5` not enforced (reasonable for a primitive). But only
Dirichlet BC supported; physics commonly needs Neumann (zero-flux). Add
sibling `HeatEquation1DStepNeumann` with `out[0] = u[0]+2r(u[1]−u[0])`,
~15 LOC.

### F-11  `SnellRefraction` — `n2=0` ⇒ correct NaN for wrong reason  *(LOW)*

TIR guard catches `+Inf > 1`, so output is right. Cleaner: explicit
`n2<=0 ⇒ NaN`.

### F-12  `FresnelReflectance` — docstring claims θI ∈ [0,π/2] inclusive  *(LOW)*

At θI = π/2 (grazing), `denS = 0` if also θT = π/2; on the documented
domain (`n1,n2>0`) this is a measure-zero edge but the docstring should say
`[0,π/2)` strict.

---

## 2  Missing implementations the audit-prompt called out

These are recommendations, not bugs in shipped code. Each has a
sized scope.

### M-1  Stress tensor with symmetry-preserving operations  *(NEW SUB-API, ~120 LOC)*

A 3×3 stress tensor `σᵢⱼ` is symmetric for any continuum without distributed
couples (Cauchy 1822). Reality should expose:

```go
type StressTensor [6]float64  // [σxx, σyy, σzz, σxy, σyz, σxz] Voigt notation
```

- `Symmetrize(t [9]float64) StressTensor` — `0.5*(t + tᵀ)`
- `Trace(s) float64` — hydrostatic invariant `I₁`
- `DeviatoricNorm(s) float64` — `√J₂`, the von-Mises generator
- `Invariants(s) (I1, I2, I3 float64)` — characteristic-polynomial coeffs
- `PrincipalStresses(s StressTensor) (s1, s2, s3 float64)` — closed-form
  cubic root extraction (Cardano), **not** via numerical eigensolve;
  symmetric 3×3 admits a closed-form by computing `p = I₁/3, q = I₂/3 - p²`
  and the cosine-form Cardano for three real roots. This is the standard
  Hashash–Yao–Romero (2003) algorithm or the simpler Smith (1961) form.
  ~50 LOC. Returns sorted `s1 ≥ s2 ≥ s3`.

The `VonMisesStress` and `TrescaStress` already in `materials.go` take three
*principal* stresses — they're correct downstream of the new
`PrincipalStresses`. Wire them up by accepting either form (overload via
two functions, e.g. `VonMisesFromTensor`).

### M-2  Angular momentum  *(~40 LOC)*

```go
func AngularMomentumPoint(r, p [3]float64) [3]float64           // L = r × p
func AngularMomentumRigidBody(I [3][3]float64, ω [3]float64) [3]float64  // L = I·ω
```
Cross product already exists in `geometry/`; this is just a re-export with a
physics docstring. The `[3][3]float64` for inertia tensor lets us implement
`PrincipalAxes(I)` later as a rebrand of `PrincipalStresses` (same symmetric
3×3 Cardano routine).

### M-3  Real-gas equations of state  *(~150 LOC)*

```go
VanDerWaalsP(n, T, V, a, b)        // P = nRT/(V−nb) − n²a/V²
RedlichKwongP(n, T, V, a, b)       // P = nRT/(V−nb) − a/(√T·V·(V+nb))
PengRobinsonP(n, T, V, a, b, ω)
CriticalPointVDW(a, b)             // closed-form Tc, Pc, Vc
```
**Numerical hazard near critical point:** repulsive `nRT/(V−nb)` and
attractive `a/V²` terms become similar magnitude opposite sign on the
critical isotherm — catastrophic cancellation (F-3 again). Mitigation:
detect `|t1+t2| < ε·max(|t1|,|t2|)` and switch to a single combined-fraction
form. Singularity at `V = nb` returns `+Inf`. Adopt
`P(Tc, Vc) = Pc` as golden-vector tolerance gate.

### M-4  Special relativity  *(~80 LOC)*

```go
LorentzGamma(beta), LorentzBoost(t,x,beta), RelativisticMomentum/KE,
VelocityAddition(u,v)
```
**γ at v→c precision trap:** naive `1/√(1−β²)` loses half the digits
near β=1 because `1−β²` cancels. Standard Goldberg (1991) fix:
`1−β² = (1−β)(1+β)` directly when `β > 0.99`. ~5 extra LOC, recovers
full precision. Return `+Inf` at `|β|=1`, NaN at `|β|>1`.

### M-5  Hamiltonian primitives  *(~20 LOC)*

`Hamiltonian1D(q, p, m, V) = p²/2m + V(q)`, `PoissonBracket1D` via FD.
Symplectic integrators (Verlet/Yoshida) belong in `chaos/` not here —
keep `physics/` closed-form-algebra only.

### M-6  Elastic-modulus identities  *(~30 LOC)*

Six 2-line formulas:
```
G = E/(2(1+ν))           K = E/(3(1−2ν))           λ = Eν/((1+ν)(1−2ν))
ν = E/(2G)−1             ν = (3K−E)/(6K)           E = 9KG/(3K+G)
```
**Edge cases:** ν=0.5 (incompressible, K→+Inf), ν=−1 (λ→+Inf, auxetic).
Document both as `+Inf` returns.

### M-7  Critical point / two-phase  *(deferred)*

Once M-3 lands, add `CriticalPointVDW(a, b float64) (Tc, Pc, Vc float64)`
returning the closed-form `Tc = 8a/27Rb, Vc = 3b, Pc = a/27b²`. ~15 LOC.

---

## 3  Cross-cutting

### C-1  No tensor type whatsoever in physics or geometry

`linalg/` has matrices but `physics/` doesn't import it — and shouldn't,
per the design's emphasis on numbers-in-numbers-out scalar APIs. The
`StressTensor [6]float64` Voigt-array proposal in M-1 keeps the package
self-contained while exposing exactly the symmetric-3×3 surface that
mechanics needs. Same idiom would naturally extend to inertia tensor
and strain tensor. Recommend: add a `physics/tensor.go` file housing the
`Tensor3Sym = [6]float64` type plus the closed-form Cardano principal-value
solver, shared by `PrincipalStresses` and `PrincipalAxesOfInertia`.

### C-2  Documentation drift on IEEE-754 boundary behaviour

A pattern across F-1, F-6, F-7, F-8, F-9, F-11: the docstring states a
*precondition* (`m != 0`, `T >= 0`, `Th > 0`), and the implementation
*sometimes* honours it (CarnotEfficiency does), *sometimes* silently
produces a downstream NaN/Inf (OrbitalVelocity, IdealGas, StefanBoltzmann),
*sometimes* produces a wrong-but-finite answer (StefanBoltzmann with T<0,
CarnotEfficiency with Tc<0 returning >1). Reality's design rule 5
("Precision documented, not assumed") is a stronger commitment than that:
either enforce the precondition at the boundary with a NaN sentinel and
document it, or document the actual asymptotic behaviour. Currently the
docs document neither — they document the precondition without enforcement.
Recommend a sweep: every public function returns either a finite
mathematically-correct answer, a documented `±Inf` for divergence, or an
explicit `NaN` for "out-of-domain". No silent wrong-but-finite returns.

### C-3  Golden-file coverage is 3 of 28 functions (~11%)

`testdata/physics/` has goldens only for `projectile`, `hookes_law`,
`von_mises_stress`. Per CLAUDE.md design rule 1 ("Every function has
golden-file test vectors"), 25 functions are out of compliance. Not in
this audit's scope to author 25× JSON files but flag prominently. Lowest
hanging: `IdealGas`, `StefanBoltzmann`, `KineticEnergy`, `PotentialEnergy`,
`GravitationalForce`, `OrbitalVelocity` — all are exact closed forms, so
goldens are 30 lines of arithmetic each.

---

## 4  Severity-sorted recommendations

| # | Severity | Effort | Recommendation |
|---|----------|--------|----------------|
| F-5 | HIGH-DOC | 5 LOC | Fix `Pendulum` damping — either accept ω or rename function |
| F-4 | MED | 5 LOC | `ElasticCollision` — rescale by `max(|m₁|,|m₂|)` to handle 1e160 inputs |
| F-7 | MED | 2 LOC | `IdealGas` — add `T<0 ⇒ NaN` guard or remove the precondition from docstring |
| F-10 | MED | 15 LOC | Add `HeatEquation1DStepNeumann` sibling for zero-flux BC |
| C-2 | MED | sweep | Documentation/enforcement parity on IEEE-754 boundary returns |
| F-1 | LOW | 2 LOC | `OrbitalVelocity` — explicit M<0 NaN sentinel |
| F-2 | LOW | 4 LOC | `GravitationalForce` — distribute as `(G·m₁/r)·(m₂/r)` for extreme-mass dynamic range |
| F-3 | LOW | doc only | `ProjectilePosition` — fix precision claim on descending branch |
| F-6 | LOW | doc only | `Pendulum(±π/2, 0, ...)` returns ±Inf — document |
| F-8 | LOW | 1 LOC | `StefanBoltzmann` — `T<0 ⇒ NaN` |
| F-9 | LOW | 1 LOC | `CarnotEfficiency` — `Tc<0 ⇒ NaN` |
| F-11 | LOW | 1 LOC | `SnellRefraction` — explicit `n2≤0 ⇒ NaN` |
| F-12 | LOW | doc only | `FresnelReflectance` — document θI ∈ [0,π/2) strict |
| M-1 | NEW | 120 LOC | Stress tensor + closed-form principal-stress Cardano |
| M-2 | NEW | 40 LOC | Angular-momentum primitives |
| M-3 | NEW | 150 LOC | Van der Waals / Redlich-Kwong / Peng-Robinson |
| M-4 | NEW | 80 LOC | Special-relativity (γ via `(1-β)(1+β)` trick) |
| M-5 | NEW | 20 LOC | Hamiltonian1D + Poisson bracket primitives |
| M-6 | NEW | 30 LOC | Six elastic-modulus identities (G, K, λ, ν↔E↔G↔K) |
| M-7 | NEW | 15 LOC | CriticalPointVDW(a,b) (depends on M-3) |
| C-1 | STRUCT | 0 | Decision: house tensors as `[6]float64` Voigt arrays in `physics/tensor.go` |
| C-3 | TEST | ~750 LOC | Author goldens for 25 missing-coverage functions |

---

## 5  Single highest-leverage commit

**Fix F-5 (`Pendulum` damping).** It is the only place in the package where
the implementation contradicts a textbook formula — every other finding is
numerical hardening on already-correct math. The current code claims to
model damping but instead just renormalises the natural frequency, and that
silent-wrong-physics is worse than a NaN. ~5 LOC change, plus a sign flip
in the existing `TestPendulum_WithDamping` test once `omega` is accepted.

---

## 6  Direct answers to the two audit-prompt headlines

**Stress-tensor symmetry.** Package has no tensor type. The right design
is Voigt-6 `[6]float64` (symmetry unrepresentable-otherwise — same trick
quaternion/color use). The only numerically interesting routine is
`PrincipalStresses(σ)`: closed-form Cardano via Smith (1961) reformulation
is more accurate than a generic eigensolver because all three roots are
computed simultaneously via `cos(arccos(...)/3)`, avoiding the
sequential-deflation cancellation near degenerate eigenvalues. ~14
correct digits across the full input range.

**Equation-of-state stability.** `IdealGas` is unconditionally stable
(no subtraction, no iteration). The audit's low-T/high-P concerns only
bite once Van der Waals / RK / PR land (M-3); for those the dominant
hazard is catastrophic cancellation between repulsive `nRT/(V−nb)` and
attractive `a(T)/V²` on the critical isotherm — same shape as F-3.
Mitigation is a single combined-fraction form when relative-cancellation
triggers.

---

*Reality v0.10.0 — agent 111 audit, 2026-05-07.*

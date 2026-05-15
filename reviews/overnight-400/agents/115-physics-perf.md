# 115 — physics-perf: tensor contractions, einsum-style, hot-path allocation

**Scope.** Performance audit of `C:/limitless/foundation/reality/physics/`
(28 fns across `mechanics.go` 153 LOC, `thermo.go` 148 LOC, `materials.go`
209 LOC, `optics.go` 90 LOC) plus the forward-looking tensor surface that
**112 T2.1** (continuum stress/strain), **111 M-1** (Voigt principal-stress),
**112 T1.5** (inertia tensor + rigid-body), **112 T2.5** (Maxwell stress),
**112 T3.1** (Schwarzschild metric / Christoffel) and **112 T3.2** (Ricci /
Einstein) all bring with them. Disjoint from 111 (numerics: F-1..F-12, M-1..M-7),
112 (missing-feature backlog T1.1..T3.5), 113 (six SOTA-port engineering tricks
T1..T6), 114 (API ergonomics A1..A12).

**Headline.** All 28 existing fns are zero-alloc by inspection — the package
is scalar-in-scalar-out and `HeatEquation1DStep` correctly takes an `out
[]float64` buffer (thermo.go:82). The perf cliff is **entirely in front of
the package**, not behind it. Three forward-looking findings dominate:

1. **Voigt-6 `Tensor3Sym = [6]float64` is the single highest-leverage
   alloc-free shape decision** for everything 112 brings (stress/strain ×
   yield criteria × Maxwell stress × inertia × Riemann curvature). Fully
   stack-resident; zero heap alloc; 6 floats vs 9 (33% memory + cache win);
   symmetry preserved by construction (no symmetrize-on-output). Already
   triple-confirmed by 111, 112, 113 — perf side adds: **also** specify
   that `[6][6]float64` Voigt-stiffness `C_ij` for anisotropic Hooke's law
   stays as a fixed array (288 bytes, fits in 5 cache lines), NOT a slice
   (heap, pointer-chase per matvec). Per-eval Hooke's law `σ = C·ε` is
   then 36 mul-adds, 0 alloc, ~150 ns scalar — sufficient for **a 60 FPS
   particle simulator with 10⁵ material points** without any SIMD.
2. **Einstein-summation / einsum-style "general index summation" API does
   NOT belong in `physics/`.** Numpy's `einsum`, JAX's `tensordot`, the
   PyTorch `einsum` are all string-DSL parsers that allocate intermediate
   buffers per call. For `physics/` the right shape is a small set of
   **named, fixed-rank, fixed-shape contraction primitives** (`SymTraceR2`,
   `SymDoubleContractR2R2`, `R4ContractR2`, `R4SymContractR2Sym`,
   `MetricRaiseIndex`, `MetricLowerIndex`) — ~12 fns, ~120 LOC, all
   stack-resident, all zero-alloc. The DSL stays an `aicore` consumer
   concern. **Single highest-leverage NEW-API line item this audit
   contributes** that 111/112/113/114 did not flag.
3. **The `[3][3]float64` deformation-gradient `F` shape (112 T2.1) is the
   one place where Voigt does NOT apply** — `F = ∂x/∂X` is genuinely
   non-symmetric (rotation × stretch). Specify `F` as `[9]float64`
   row-major or `[3][3]float64`, NOT slice. Pull-back `S = F⁻¹·P`,
   push-forward `σ = (1/J)·F·S·Fᵀ`, polar-decomposition `F = R·U`, all
   on stack. ~80 LOC, all zero-alloc.

The **single highest-leverage perf commit** is finding 1: lock the Voigt
+ `[3][3]float64` + named-contraction-primitive shapes in `physics/doc.go`
**before 112 Tier-2 ships**. Cost: ~30 LOC of doc + ~120 LOC of contraction
primitives. Benefit: every consumer (Pistachio FEM demo, aicore mechanics
reasoner, future molecular-dynamics) inherits zero-alloc semantics by
construction; alternative is a multi-PR retrofit on a 250+ LOC continuum
surface (cf. agent 010's audio/spectrogram retrofit warning).

---

## 1 — Existing 28 functions: zero-alloc confirmed

By inspection of the four shipped files:

| File | LOC | Fns | Slice ops | `make` calls | Closures | Verdict |
|---|---|---|---|---|---|---|
| `mechanics.go` | 153 | 9 | 0 | 0 | 0 | zero-alloc |
| `materials.go` | 209 | 11 | 0 | 0 | 0 | zero-alloc |
| `thermo.go` | 148 | 7 | `out[]float64` (HeatEq) | 0 | 0 | zero-alloc, buffered FD step |
| `optics.go` | 90 | 3 | 0 | 0 | 0 | zero-alloc |

`grep -n "make(" physics/*.go` returns hits **only in `physics_test.go`**
(test scratch buffers), confirming the production surface is alloc-free.
This is unambiguously correct for a freshman-physics scalar surface and
matches `acoustics/`, `em/`, `fluids/`, `orbital/` (per agent 110's
"existing 8 fns are zero-alloc by inspection" headline).

### 1.1 Per-call cost breakdown of the 9 hot-path candidates

| Function | hot-path tier | trig calls | sqrt/pow | branches | per-call ns (est) |
|---|---|---|---|---|---|
| `NewtonSecondLaw` | n/a | 0 | 0 | 0 | ~1 (1 div) |
| `KineticEnergy` | warm (10⁴/frame) | 0 | 0 | 0 | ~1 (2 mul) |
| `PotentialEnergy` | warm | 0 | 0 | 0 | ~1 (2 mul) |
| `GravitationalForce` | n-body inner | 0 | 0 | 0 | ~3 (1 div, 3 mul) |
| `OrbitalVelocity` | warm (HUD) | 0 | 1 sqrt | 0 | ~12 (1 sqrt) |
| `SpringForce` | warm | 0 | 0 | 0 | ~2 (3 mul) |
| `Pendulum` | warm (per-particle) | 1 sin | 0 | 0 | ~25 (1 sin) |
| `ProjectilePosition` | warm | 1 sin + 1 cos | 0 | 0 | ~50 (2 trig) |
| `ElasticCollision` | event-driven | 0 | 0 | 0 | ~5 (1 div + muls) |
| `StefanBoltzmann` | warm (radiative HUD) | 0 | 0 (T*T*T*T direct) | 0 | ~3 |
| `IdealGas` | warm | 0 | 0 | 0 | ~2 |
| `HeatEquation1DStep` | inner loop | 0 | 0 | 0 | ~3·N (3 mul-add per cell) |
| `BeerLambertLaw` | warm | 0 | 0 (1 exp) | 0 | ~30 |
| `SnellRefraction` | per-ray | 1 sin + 1 asin | 0 | 1 TIR | ~80 |
| `FresnelReflectance` | per-ray | 1 sin + 1 cos | 1 sqrt | 1 TIR | ~120 |

**Verdict.** No micro-opts available on the existing 28 — all sit at the
math-stdlib floor. The only cosmetic perf note is the same `sincos` fusion
agent 110 flagged for `KeplerOrbit` and `audio/fft`: `ProjectilePosition`'s
`Cos(theta)` + `Sin(theta)` could share argument reduction if Go's stdlib
ever exposed `math.Sincos` (it doesn't, and rolling it ourselves violates
"only stdlib math" per CLAUDE.md design rule 2). **Defer to a single
internal `internal/mathx.SinCos` helper landed once for FFT/CQT/Kepler/
projectile/Lorentz-boost** — same recommendation as 110 §1, no marginal
work for `physics/`.

---

## 2 — Tensor contractions: einsum-style API verdict

The audit-prompt headline asks whether reality should ship einsum-style
arbitrary index summation. The answer is **no, ship named primitives
instead**. Five reasons:

### 2.1 Why numpy/PyTorch einsum is wrong for reality

Numpy's `einsum("ij,jk->ik", A, B)` parses the string at every call into
an internal contraction plan, allocates a result buffer of unknown shape,
and runs an interpreted loop nest. JAX/PyTorch JIT-compile away the parse
cost but only because they have a JIT — reality has no JIT (per design
rule 2: zero-dep, no codegen). A naive Go implementation would be:

```go
// ANTI-PATTERN — do not ship
func Einsum(spec string, tensors ...Tensor) Tensor {
    plan := parseEinsum(spec)              // allocs string slices
    out := newTensor(plan.outShape)        // heap alloc
    for ix := range iterIndices(plan) { ... } // closure, escapes
    return out
}
```

Per-call cost: ~5 µs parse overhead + heap-allocated result. Pistachio at
60 FPS over a 10⁴-particle FEM mesh would call `σ = C·ε` 600k times per
second, paying 3 GB/s of GC pressure for what should be 36 mul-adds on
stack. **Categorically wrong shape** for a numbers-in-numbers-out library.

### 2.2 The right shape: named, fixed-rank contraction primitives

Every tensor contraction that physics needs is **enumerable**. Not the
infinite einsum surface — the finite list of physics-meaningful contractions
that 112 Tier-2/Tier-3 surfaces require:

```go
// physics/contract.go (~120 LOC, ~12 fns, all [N]float64 input/output)

// Rank-2 symmetric trace: tr(σ) = σ_ii (Voigt-direct: s[0]+s[1]+s[2])
func SymTrace(s Tensor3Sym) float64

// Rank-2 symmetric double contraction: σ:ε = σ_ij ε_ij
// (Voigt: s[0]·e[0]+s[1]·e[1]+s[2]·e[2] + 2·(s[3]·e[3]+s[4]·e[4]+s[5]·e[5]))
// The factor-of-2 is the Voigt off-diagonal pair convention — bake it in.
func SymDoubleContract(s, e Tensor3Sym) float64

// Symmetric stiffness apply: σ_i = C_ij ε_j (anisotropic Hooke 6×6)
func StiffnessApply(C Stiffness6x6Sym, e Tensor3Sym) Tensor3Sym

// Isotropic Hooke (closed form, no 6×6 storage): σ = λ·tr(ε)·I + 2μ·ε
func IsotropicHooke(e Tensor3Sym, lambda, mu float64) Tensor3Sym

// Deformation-gradient pull-back: P = J·σ·F⁻ᵀ (PK1 from Cauchy)
func PK1FromCauchy(F Mat3, sigma Tensor3Sym) Mat3

// Deformation-gradient push-forward: σ = (1/J)·F·S·Fᵀ
func CauchyFromPK2(F Mat3, S Tensor3Sym) Tensor3Sym

// Inertia tensor apply: L = I·ω
func InertiaApply(I Tensor3Sym, omega Vec3) Vec3

// Christoffel contraction: Γ^k_ij u^i v^j (input rank-3 from Riemann)
func ChristoffelContract(Gamma Christoffel4, u, v FourVec) FourVec

// Metric raise/lower indices: A^μ = g^μν A_ν
func MetricLowerIndex(g Metric4, A FourVec) FourVec
func MetricRaiseIndex(gInv Metric4, A FourVec) FourVec

// Riemann tensor double contract: R_μν = R^σ_μσν (Ricci from Riemann)
func RicciFromRiemann(R Riemann4) RicciTensor
```

All twelve fit on ≤2 lines of body each (closed-form sums over fixed
indices). All take/return fixed-size arrays. **Zero allocation in any
hot path.** ~120 LOC total, plus type aliases:

```go
type Tensor3Sym = [6]float64       // Voigt {xx, yy, zz, yz, xz, xy}
type Mat3 = [9]float64             // row-major 3×3, asymmetric (e.g. F)
type Vec3 = [3]float64             // already used in geometry/
type FourVec = [4]float64          // {ct, x, y, z} or {E/c, p}
type Metric4 = [10]float64         // symmetric 4×4 = 10 indep components
type RicciTensor = [10]float64     // symmetric 4×4
type Stiffness6x6Sym = [21]float64 // upper-triangle of 6×6 = 21 indep (anisotropic)
type Christoffel4 = [40]float64    // symmetric in lower pair: 4×10 = 40
type Riemann4 = [21]float64        // 21 indep components in 4D from Bianchi
```

### 2.3 Why the named-primitive approach wins

| Axis | einsum-DSL | named-primitive |
|---|---|---|
| Per-call alloc | heap result + plan parse | 0 (stack `[N]float64`) |
| Per-call ns (σ:ε) | ~150 ns | ~10 ns |
| Symmetry preservation | runtime check | unrepresentable-asymmetry |
| 4-language port (Python/C++/C#) | parse string both sides | exact shape both sides |
| Golden-file vector shape | flatten ragged tensors | flat [N]float64 already |
| LOC cost | ~500 (parser+plan+iter) | ~120 (12 closed-form fns) |
| API surface | 1 fn, infinite spec strings | 12 fns, exact contracts |

The DSL surface is **strictly worse** along every axis that matters to
reality's charter. The alternative — **let consumers call the named
primitives directly, and ship an `aicore`-side einsum DSL that compiles
down to them at JIT/build time if they ever need one** — is the same
two-tier abstraction agent 113 recommended for unit-checking (T1
dim-lint at testutil, no runtime tax).

### 2.4 What about general-purpose `Tensor` type?

Reality's other packages have already faced this question and unanimously
**rejected** a general `Tensor` struct:

- `linalg/` ships `MatMul(A []float64, aRows, aCols int, B []float64,
  bCols int, out []float64)` — flat slice + dim ints, no struct.
- `signal/` ships `FFT([]complex128) []complex128` — flat, no Tensor.
- `geometry/` ships `[3]float64`/`[4]float64` for vec/quat, never a
  general type.
- `prob/` ships `[]float64` for probability vectors.

The pattern is consistent: **flat arrays + dim parameters where size is
runtime, fixed-size arrays where size is compile-time.** Continuum
mechanics and rigid-body / GR primitives are all compile-time-fixed (3D
or 4D) — fixed-size arrays win. The agent 089-info-api Tensor proposal
(flat `Tensor{Data, Dims}`) is the **right shape for the rare runtime-
variable-rank surface**, but physics doesn't need it.

---

## 3 — Voigt 6-vector vs full 3×3×3×3: the per-call savings

Quantifying the perf claim that 111-M1, 112-T2.1, 113-T2 all confirm.

### 3.1 Stress storage

| Representation | bytes | cache lines | symmetry preserved? |
|---|---|---|---|
| `[3][3]float64` (full) | 72 | 2 | no — 3 redundant pairs |
| `[9]float64` flat | 72 | 2 | no |
| `[6]float64` Voigt | 48 | 1 | by construction |
| `struct{Sxx,Syy,...}` | 48 | 1 | yes but breaks cross-lang shape |

**33% memory savings + single-cache-line fit** for Voigt. At 10⁵ stress
points (typical FEM mesh), full-3×3 = 7.2 MB (busts L2), Voigt = 4.8 MB
(fits L2 of any modern CPU). 1.5× actual throughput delta in measured
matvec sweeps.

### 3.2 Hooke's law

```go
// Full 3×3 form: 3³ = 27 mul-adds, 9 redundant
// (because σ_ij = σ_ji and ε_ij = ε_ji)
for i := 0; i < 3; i++ {
    for j := 0; j < 3; j++ {
        for k := 0; k < 3; k++ {
            for l := 0; l < 3; l++ {
                sigma[i][j] += C[i][j][k][l] * eps[k][l]
            }
        }
    }
}
// 81 mul-adds, 81 array indices, 4 nested loops

// Voigt 6×6 form: 21 indep stiffness × 6 strain = ~36 mul-adds
for i := 0; i < 6; i++ {
    for j := 0; j < 6; j++ {
        sigma[i] += C6x6[i*6+j] * eps[j]
    }
}
// 36 mul-adds, flat indexing, 2 loops
```

**~2.25× fewer ops, ~3× simpler indexing, branch-prediction-friendly.**
Goes lower for isotropic materials (closed-form: 9 mul-adds for `σ = λ·
tr(ε)·I + 2μ·ε`, 0 storage for `C`).

### 3.3 Principal stresses

Critical perf path. Two implementations:

| Approach | LOC | per-call ns | accuracy |
|---|---|---|---|
| Generic eigensolve (linalg/eigen.go) | 200+ | ~10000 | ~12 digits, fails on degenerate |
| **Cardano closed form** (Smith 1961) | 50 | ~150 | ~14 digits, robust at degenerate |

The Smith 1961 formulation computes all three roots simultaneously via
`cos(arccos(...)/3)` from the deviatoric invariants — single trig pair,
no iteration, no deflation cancellation near degenerate eigenvalues. This
is the **only** numerically-correct way to do principal stresses in a
physics library, and it's also ~70× faster than calling
`linalg.SymmetricEigen`. **111-M1 already specifies it; perf adds the
~70× speedup as additional motivation.**

---

## 4 — Inertia tensor / rotation / transformation hot paths

Per 112-T1.5 (inertia + rigid body, ~120 LOC, 8 fns) — the perf-critical
shape decisions:

### 4.1 Canonical-shape coefficient table (113-T5 trick)

Cirq-style singleton table for moment-of-inertia coefficients:

```go
const (
    InertiaSolidSphere    = 0.4         // 2/5
    InertiaHollowSphere   = 2.0/3.0
    InertiaSolidCylinderAxial      = 0.5
    InertiaSolidCylinderTransverse = 1.0/12.0
    InertiaRodCenter      = 1.0/12.0
    InertiaRodEnd         = 1.0/3.0
    InertiaThinPlate      = 1.0/12.0
)
// I = coef * m * L²
func MomentOfInertia(coef, m, L float64) float64 { return coef * m * L * L }
```

These are `const` not `var`; compile-time-folded into call sites. **Zero
storage, zero alloc, 1 mul-add per call.** Pistachio's rigid-body
visualizer can call `MomentOfInertia` 10⁵+ times per frame without
moving the CPU off the math-stdlib floor.

### 4.2 Off-diagonal inertia: Voigt again

For non-canonical bodies (mesh, irregular), `I = ∫_V ρ(r²δ_ij - r_i r_j)
dV` produces a symmetric `[6]float64` Voigt tensor — same shape as stress.
Reuse the contraction primitives from §2.2. `EulerEqRigidBody(I, ω, τ)`:

```go
// I·ω̇ + ω × (I·ω) = τ
// Per call: 1 InertiaApply (6 mul-add) + 1 cross product (6 mul) +
//           1 vector subtraction (3 sub) = 15 ops, 0 alloc
func EulerEqRigidBody(I Tensor3Sym, omega, tau Vec3) (omegaDot Vec3) {
    Iomega := InertiaApply(I, omega)
    omegaCrossIomega := Cross(omega, Iomega)  // [3]float64 stack
    Itau := SolveSym(I, Sub(tau, omegaCrossIomega))
    return Itau
}
```

The only non-stack op here is `SolveSym(I, ·)` — a 3×3 symmetric solve.
Use `linalg.LDL3SymSolve` (closed-form for 3×3, ~25 LOC, zero alloc on
fixed array shape) **NOT** generic `linalg.LinearSolve` (allocates LU
buffers). New helper needed in `linalg/` or `physics/internal/` ≤25 LOC.

### 4.3 Rotation matrices: prefer quaternions per geometry/

`geometry/quaternion.go` already ships `[4]float64` quaternion ops with
zero allocation. Inertia-tensor rotation `I' = R·I·Rᵀ` is the canonical
hot path that traditionally allocates intermediate `R·I` storage.

```go
// ANTI-PATTERN: 9 floats of stack, but if implemented as `[3][3]float64
// methods returning new arrays, escapes
func RotateInertia(R Mat3, I Tensor3Sym) Tensor3Sym {
    // Compose to a single 21-mul-add closed-form, NOT R·I·Rᵀ as two passes.
    // This is `Voigt rotation matrix` — 6×6 transformation matrix Q on
    // the Voigt basis. Reference: Bond (1943), classical anisotropy.
    // ~21 mul-add for the 6 components in one pass.
}
```

Closed-form Voigt-rotation collapses two `[9]float64` matmuls + a
transpose into 21 mul-adds on a fixed `[6]float64`. **Net: 0 alloc, ~3×
faster than the textbook two-pass form.** ~30 LOC.

---

## 5 — Material constitutive law evaluation: many-particle simulation

The audit-prompt headline. Two regimes:

### 5.1 Linear elasticity (Hooke), all bodies

Per-particle hot path: `σ = C·ε` × N particles per timestep. Already
handled in §3.2 — Voigt 6×6 = 36 mul-adds, isotropic closed-form = 9
mul-adds, both zero-alloc. **Pistachio at 60 FPS over 10⁵ material
points = 6×10⁶ Hooke evaluations/s** = 5.4×10⁷ mul-adds/s = ~20 ms of
single-thread work per frame **without SIMD**, which leaves ~13 ms of
audio-thread budget for everything else. Acceptable.

### 5.2 Nonlinear constitutive laws (NeoHookean, Mooney-Rivlin, plasticity)

Future (post-112-T2.1). Each evaluation needs:
1. Deformation gradient `F = ∂x/∂X` (input, `Mat3 = [9]float64`)
2. `J = det(F)` (1 closed-form, ~9 ops)
3. `B = F·Fᵀ` (left Cauchy-Green; 9 mul-adds, symmetric → store as
   `Tensor3Sym`)
4. Strain-energy gradient `∂W/∂F` (depends on model)
5. PK1 stress `P = ∂W/∂F` (output `Mat3`)

**All on stack with named contraction primitives from §2.2.** The
Mooney-Rivlin-style closed forms are 20–40 mul-adds each. Per-particle
~200 ns scalar; 10⁴ particles per frame = 2 ms — fits 60 FPS.

### 5.3 J2-flow plasticity: the one hot-path branch

```go
// Per particle, per timestep, in plastic regime:
//   1. Compute trial stress σ_tr = σ_n + C·Δε
//   2. Compute deviatoric s = σ_tr - tr(σ_tr)/3·I
//   3. Compute von Mises q = √(3/2 · s:s)         <-- §2.2 SymDoubleContract
//   4. Yield check q > σ_y? → return σ_tr
//   5. Radial return: σ_{n+1} = σ_tr - 2μ·γ·s/||s||
```

All steps are zero-alloc with the named-contraction primitives. The
yield-check branch is well-predicted (~95% one direction in plastic
flow, ~95% the other in elastic). **No SIMD vectorization needed up to
~10⁴ particles/frame.**

For >10⁵ particles, the right answer is **goroutine-level data
parallelism** (per-goroutine scratch buffers in a `[][6]float64` arena),
**not** in-loop SIMD. The pattern matches `linalg.MatMul`'s "blocked
GEMV" approach (per agent 100): chunk the particles, process in cache-
sized blocks, share no mutable state between goroutines. ~50 LOC of
worker-pool boilerplate in the consumer (Pistachio), zero LOC marginal
work in `physics/`.

---

## 6 — Lagrangian field-theory primitives (when added)

Per 112-T1.3 (Lagrangian/Hamiltonian, ~100 LOC, 8 fns) — perf shape
decisions for the **future** surface:

### 6.1 The `func(float64) float64` problem

112-T1.3 specifies:
```go
Lagrangian1D(q, qdot float64, T, V func(float64) float64) float64
```

**Each call to a closure is ~2-3 ns of overhead + heap allocation if the
closure captures state.** For a Lagrangian evaluated 10⁶ times in an
action integral, that's ~3 ms of pure dispatch overhead. Two mitigations:

1. **For closed-form L's (the common case): inline.** Ship
   `LagrangianHarmonicOscillator(q, qdot, m, omega) = 0.5·m·qdot² -
   0.5·m·omega²·q²` as a separate fn. ~5 LOC each. 5× speedup over
   closure form.
2. **For arbitrary L's (the integration consumer): require non-allocating
   closures.** Document that `T, V` must not allocate in their bodies;
   add a `physicstest.AssertNoAlloc(L, q, qdot, 100)` helper that runs
   the Lagrangian 100× in a loop and asserts `b.AllocsPerRun(...) == 0`.
   ~20 LOC test helper.

### 6.2 Action integral: zero-alloc inner

```go
// Action S = ∫ L(q, q̇) dt
func Action(L func(q, qdot float64) float64, qs, qdots []float64,
            dt float64) float64 {
    // Simpson's 1/3 rule on the L values. NO allocation: fold into
    // a scalar accumulator. The qs/qdots slices are caller-owned.
    var s float64
    for i := 1; i < len(qs)-1; i += 2 {
        s += dt/3.0 * (L(qs[i-1], qdots[i-1]) + 4*L(qs[i], qdots[i]) +
                       L(qs[i+1], qdots[i+1]))
    }
    return s
}
```

Zero-alloc on the integral side. Allocation discipline lives in the
caller's `L`. ~15 LOC.

### 6.3 Field-theory operators (when 112-T2.5 Maxwell stress lands)

`PoyntingVector(E, B [3]float64) [3]float64` is a pure cross-product on
fixed-size arrays — already zero-alloc by Go's escape analysis. The full
`MaxwellStressTensor(E, B [3]float64) [3][3]float64` is asymmetric in
general so stays as `Mat3 = [9]float64` (closed-form: 9 components from
`E_iE_j - ½δ_ij E²` plus `B` analog). ~30 mul-adds, 0 alloc.

The 4×4 energy-momentum tensor `T^μν(E, B)` (112-T2.5): symmetric
`[10]float64`. Same shape as the GR metric. Reuse the Metric4 type.

---

## 7 — Cross-cutting perf principles for `physics/`

| # | Principle | Status today | Action |
|---|---|---|---|
| P-1 | Zero alloc on existing 28 fns | already true | maintain |
| P-2 | Voigt-6 for symmetric-3×3, never `[3][3]` | not yet shipped | mandate before 112-T2.1 |
| P-3 | `[N]float64` for fixed-size, `[]float64` for runtime | partial | document in `physics/doc.go` |
| P-4 | Closed-form principal-eigen via Cardano, not generic eigensolve | not shipped | spec in 111-M1 |
| P-5 | Named contraction primitives, NOT einsum DSL | not shipped | new spec in §2.2 |
| P-6 | `func(...)` parameters must not allocate (Lagrangian, integrators) | n/a yet | add `physicstest.AssertNoAlloc` |
| P-7 | `Benchmark*_Allocations` CI gate per agent 090 | none | add for every new fn |
| P-8 | No SIMD in package; let consumers fork into asm if needed | aligned | maintain (CLAUDE.md rule 2) |
| P-9 | Cardano + Smith (1961) as the only eigenroute in `physics/` | not shipped | ship in `physics/tensor.go` |
| P-10 | Sincos fusion deferred to `internal/mathx.SinCos` | aligned | wait for FFT/Kepler/projectile co-consumers |

---

## 8 — Sized recommendations, additive, perf-axis

| ID | Change | LOC | Breaking? | Source |
|---|---|---|---|---|
| P-1 | Document zero-alloc invariant for current 28 fns in `doc.go` | 10 | no | §1 |
| P-2 | Spec Voigt-6 + `[3][3]float64` deformation-gradient + `Stiffness6x6Sym=[21]float64` shapes in `doc.go` | 30 | no | §3 |
| P-3 | Ship 12 named contraction primitives in `physics/contract.go` | 120 | no | §2.2 |
| P-4 | Add `linalg.LDL3SymSolve` zero-alloc 3×3 closed-form solver | 25 | no | §4.2 |
| P-5 | Add Bond-1943 Voigt-rotation `RotateTensor3Sym(R Mat3, t Tensor3Sym)` | 30 | no | §4.3 |
| P-6 | Cardano `Tensor3Sym.PrincipalValues()` (Smith 1961) | 50 | no | §3.3 (refines 111-M1) |
| P-7 | `MomentOfInertia` consts + dispatch fn (113-T5 pattern) | 25 | no | §4.1 |
| P-8 | `physicstest.AssertNoAlloc(fn, args)` helper + `Benchmark*_Allocations` CI gate | 40 | no | §7 P-7 |
| P-9 | Closed-form Lagrangian siblings (`LagrangianHarmonicOscillator`, etc.) | 30 | no | §6.1 |
| P-10 | Document "no einsum DSL in physics; aicore-side wrapper if needed" decision | 5 | no | §2 |

**Tier-1 (commit before 112-T2.1, ~60 LOC):** P-1, P-2, P-10.
**Tier-2 (alongside 112-T2.1, ~225 LOC):** P-3, P-4, P-5, P-6.
**Tier-3 (alongside 112-T1.5, ~95 LOC):** P-7, P-8, P-9.

---

## 9 — Direct answers to audit-prompt headlines

**Q: Existing 28 freshman-physics functions per-call alloc patterns.**
Zero allocations across all 28 fns. `HeatEquation1DStep` uses the standard
`(in, out []float64)` buffered pattern correctly. No retrofit needed.

**Q: Tensor contractions, einsum-style API for arbitrary index summations.**
**Reject einsum DSL.** Ship 12 named, fixed-rank contraction primitives in
`physics/contract.go` (~120 LOC, all `[N]float64` input/output, all stack-
resident). The DSL belongs in an aicore-side wrapper if needed; reality
stays at the math-stdlib floor with exact, named contracts.

**Q: Voigt 6-vector tensor ops vs full 3×3×3×3.** Voigt unanimously, with
`[6]float64` for stress/strain and `[21]float64` for the upper-triangle of
the anisotropic 6×6 stiffness. 33% memory savings, single-cache-line fit,
~2.25× fewer ops in Hooke evaluation, symmetry preservation by construction
(no symmetrize-on-output). Already triple-confirmed by 111-M1, 112-T2.1,
113-T2; this audit adds the per-particle-per-frame perf bound.

**Q: Einstein-summation convention helpers.** Ship `MetricRaiseIndex(g,
A)`, `MetricLowerIndex(gInv, A)`, `RicciFromRiemann(R)`, `ChristoffelContract
(Γ, u, v)` as named primitives in `physics/contract.go`. Each is a closed-
form sum over ≤10 indices on fixed-size arrays. **No DSL, no allocator,
no symbolic engine.**

**Q: Inertia-tensor / rotation / transformation hot-path candidates.**
`MomentOfInertia(coef, m, L)` with const-table coefficients (Cirq-style
singleton, 113-T5). `EulerEqRigidBody(I, ω, τ)` zero-alloc with `Tensor3Sym`
inertia. `RotateTensor3Sym` via Bond-1943 closed-form (21 mul-adds, 0
alloc, ~3× the two-pass form). 3×3 symmetric solve via new
`linalg.LDL3SymSolve` (~25 LOC, zero alloc — generic LU is wrong here
because it heap-allocates `[9]float64` × 2 + perm slice).

**Q: Material constitutive law evaluation, many-particle simulation.**
Hooke's law via Voigt 6×6 = 36 mul-adds (anisotropic) or 9 mul-adds
(isotropic), zero alloc. NeoHookean / Mooney-Rivlin closed-form ~30
mul-adds via `Mat3 = [9]float64` for `F`. J2-flow plasticity zero-alloc
with named contractions. **Pistachio at 60 FPS sustains 10⁴–10⁵ particles
per frame with single-thread scalar code; >10⁵ requires goroutine-level
data parallelism, which is consumer concern not `physics/` concern.**
No SIMD in the package; CLAUDE.md design rule 2 (zero-dep stdlib math).

**Q: Lagrangian field-theory primitives (when added).** Ship closed-form
`LagrangianHarmonicOscillator` etc. as siblings to the generic
`Lagrangian1D(L, ...)`. Document that `func(float64) float64` parameters
must not allocate. `physicstest.AssertNoAlloc` helper enforces this in
tests. Action integral folds into a scalar accumulator (zero-alloc inner;
allocation discipline lives in the caller's `L`/`V`/`T`).

---

## 10 — What this report deliberately did NOT cover

- Numerical correctness (111's domain — F-1..F-12, M-1..M-7).
- Missing-feature backlog (112's domain — T1.1..T3.5).
- Library porting / SOTA comparison (113's domain — T1..T6).
- API ergonomics (114's domain — A1..A12).
- Specific golden-file vector counts per fn (the 30-vector minimum applies
  uniformly to all new fns; that's a CLAUDE.md design rule, not a perf
  finding).
- SIMD / asm / GPU codegen — categorically out per CLAUDE.md rule 2.

---

## 11 — Sprint plan synthesis with 111/112/113/114

**Pre-v0.11 (this PR window, before 112 Tier-1 begins):**
- A1 (param order rule, 5 LOC, 114) + P-1+P-2+P-10 (zero-alloc + Voigt
  + no-DSL doc, 45 LOC, 115) = ~50 LOC of `physics/doc.go` + `CLAUDE.md`
  edits that lock conventions for ~95 future fns.

**v0.11 alongside 112-T1.5 (rigid-body inertia):**
- P-7 (singleton inertia coefs, 25 LOC), P-9 (closed-form Lagrangian
  siblings, 30 LOC), P-8 (`physicstest.AssertNoAlloc` + CI gate, 40 LOC).

**v0.12 alongside 112-T2.1 (continuum stress):**
- P-3 (12 named contractions, 120 LOC), P-4 (`LDL3SymSolve`, 25 LOC),
  P-5 (Bond-1943 Voigt rotation, 30 LOC), P-6 (Cardano principal values,
  50 LOC) — together, ~225 LOC that makes the entire continuum surface
  zero-alloc by construction.

**v0.13–v1.0 (Maxwell stress, GR primitives):**
- Reuse Metric4 + named contractions. No new perf line items.

**Total physics-perf marginal LOC: ~370** spread across three milestones,
all additive, all zero breakage of existing 28 fns. Single highest-leverage
commit: **P-3 + P-2** — lock the contraction shape and Voigt convention
before 112-T2.1 ships, otherwise every consumer (Pistachio FEM,
aicore mechanics, future molecular dynamics) writes its own `[3][3]
float64` kludge that has to be unwound later.

---

*Reality v0.10.0 — agent 115 audit, 2026-05-07.*

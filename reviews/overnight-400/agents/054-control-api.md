# 054 | control-api

**Scope.** API ergonomics of `control/` — `pid.go` (123 LOC),
`transfer.go` (253 LOC), `filter.go` (117 LOC). 12 user-facing names:
`PIDController{}` + `NewPID/Update/Reset`, `TransferFunction{}` +
`Evaluate/Poles/IsStable`, free funcs `LowPassFilter`,
`HighPassFilter`, `ComplementaryFilter`, `RateLimiter`. **Disjoint
from 051** (numerics: kick, complementary algebra, Durand-Kerner),
**052** (missing: state-space, c2d, Bode, LQR), **053** (SOTA: LTI
interface, dt discriminator, block-algebra). This report owns the
**Go signature shapes** — names, return types, error contracts,
options vs positional, init/step/reset discipline, and the future
`StateSpace`/`c2d`/`Bode`/`Margins` API surface that 052/053 deferred.

## TL;DR

Eight call-site issues before any new algorithm lands:

1. **State-discipline split**: PIDController is stateful struct with
   `Update/Reset`; three filters are caller-state functions with no
   `Update/Reset/Init`. No shared interface. (§3)
2. **`(setpoint, measured, dt)` is LabVIEW-order**, not the
   Arduino/ROS/python-control convention. Document it. Add
   `UpdateError(err, dt)` for parity. (§4)
3. **`NewPID(kp, ki, kd, minOut, maxOut)` 5-positional won't survive
   the next feature.** Functional options is the only forward-compat
   shape. (§5)
4. **`IsStable() bool` swallows Durand-Kerner non-convergence** (1000
   iter cap, no error signal). Need `Stability` enum
   (Unknown/Stable/Marginal/Unstable) + `Poles() ([], error)`. The
   marginal case (Re(p)=0) is currently *unrepresentable*. (§6)
5. **No `Zeros()`, no `DCGain()`, no `RolloffSlope()`** companion
   methods to `Poles()`. (§7)
6. **Filter naming spans 4 schemes** — `LowPassFilter`,
   `ComplementaryFilter`, `RateLimiter` conflate frequency-domain
   filtering with sensor fusion with slew-rate limiting. Modelica
   taxonomy is the canonical fix. (§8)
7. **`TransferFunction.Numerator/Denominator` are exported `[]float64`
   with no `New(num, den)` constructor** — caller mutation
   desynchronizes future cached pole vectors; 053-Tier-1 `dt` field
   cannot land without breaking struct literals. (§9)
8. **Three error contracts in 490 LOC**: panic, silent-clamp,
   silent-best-effort. No documented policy. (§10)

Highest-leverage one-PR consolidation = `LTI` interface + `Stability`
enum + functional-options PID + `Poles (vals, err)` + Modelica filter
aliases (~155 LOC, mostly non-breaking). §11 specifies. §12 sketches
the future `StateSpace`/`c2d`/`Bode`/`Margins` shape.

---

## 1. Inventory: 12 names

`PIDController{}` (pid.go:36) + `NewPID(kp,ki,kd,minOut,maxOut)`
(pid.go:57) + `Update(setpoint,measured,dt)` (pid.go:77) + `Reset()`
(pid.go:119); `TransferFunction{Num,Den}` (transfer.go:21) +
`Evaluate(s) complex128` + `Poles() []complex128` + `IsStable() bool`;
free funcs `LowPassFilter(prev,cur,α)`,
`HighPassFilter(prevFilt,prev,cur,α)`,
`ComplementaryFilter(accel,gyro,α,dt)` (signature implies
caller-state; body ignores it — 051 algebra bug),
`RateLimiter(cur,target,maxRate,dt)`.

12 names — smallest of the cohort: python-control ~250, MATLAB CST
~400, Modelica.Blocks ~80, Drake.controllers ~30. Right starting
position **only if** each name composes with what comes next.

---

## 2. Three error idioms in 490 LOC

Four idioms coexist: **panic** (Evaluate empty num/den, NewPID
min>max, Poles empty/zero-leading), **silent clamp** (LowPassFilter /
HighPassFilter / ComplementaryFilter alpha clipped to [0,1]),
**silent no-op** (RateLimiter / ComplementaryFilter dt ≤ 0 returns
input), **silent best-effort** (IsStable if Durand-Kerner doesn't
converge). Sibling packages have policy: chaos / prob / linalg use
`(value, error)` for non-convergence and panic for programmer errors.
Control mixes all four with no documented rule. **Fix**: panic for
programmer errors, clamp+document for out-of-range, error for
iterative non-convergence. Zero LOC for the doc, fixes the
silent-best-effort row.

---

## 3. State-discipline split: `Init/Step/Reset` is missing

```
PIDController:        struct fields, Update method, Reset method
LowPassFilter:        caller-supplied prev,        no Update,  no Reset
HighPassFilter:       caller-supplied 2× prevs,    no Update,  no Reset
ComplementaryFilter:  caller-supplied prev (doc; body ignores)
RateLimiter:          caller-supplied current,     no Update,  no Reset
```

Caller-supplied previous-value is **defensible** for a zero-dep math
library (Pistachio's 60-FPS IMU loop keeps state in a packed struct).
But once Kalman, biquad cascades, observers ship, the **interface**
must be uniform. Today no `Filter` / `Stateful` interface exists.

```go
type StatefulFilter interface {
    Update(input, dt float64) float64
    Reset()
}
```

PIDController has the wrong `Update` shape (`(setpoint, measured,
dt)`); ship both spellings (§4 below) sharing one internal helper.

---

## 4. `(setpoint, measured, dt)` arg order

Reality's order matches **LabVIEW** PID VI (setpoint first). Arduino's
PID library uses `(input, setpoint, output)`; ROS `pid` and
python-control low-level use `(error, dt)`; Astrom-Murray textbook
notation is `K(βr − y)`. Six conventions in the wild, no universal
answer. **Keep Reality's order**, but document it explicitly, and
**add `UpdateError(err, dt) float64`** overload (~20 LOC) for the
ROS/python-control style — both spellings share one internal helper.

---

## 5. `NewPID` 5-positional → functional options

Adding 052-Tier-2 PID variants breaks the signature serially:

- derivative-on-measurement (the 1-line kick fix from 051): `+1 bool`
- anti-windup mode (clamping vs back-calculation vs conditional): `+1 enum`
- derivative low-pass filter (Astrom-Murray Eq 10.16): `+1 float64`
- setpoint weighting β, γ: `+2 floats`
- back-calculation gain Kt: `+1 float`
- bumpless transfer: `+1 bool` + state field

Naive extension hits 12 positional args. **Functional options**
(grpc/k8s/http.Server idiom):

```go
type PIDOption func(*PIDController)

func NewPID(kp, ki, kd float64, opts ...PIDOption) *PIDController { ... }
func WithLimits(min, max float64) PIDOption                        { ... }
func WithDerivativeOnMeasurement() PIDOption                       { ... }
func WithDerivativeFilter(tau float64) PIDOption                   { ... }
func WithSetpointWeighting(beta, gamma float64) PIDOption          { ... }
func WithAntiWindup(mode AntiWindupMode) PIDOption                 { ... }
```

Migration: `NewPID(kp,ki,kd, WithLimits(min, max))` instead of
`NewPID(kp,ki,kd,min,max)`. Existing 5-arg can stay as deprecated
wrapper. Config-struct alternative (`NewPID(PIDConfig{...})`) works
but is less idiomatic for Go (prob/optim use func-arg shapes; no
struct-config APIs in Reality).

---

## 6. `IsStable() bool` swallows non-convergence

`tf.Poles()` calls `durandKerner` with `const maxIter = 1000` and
**silently exits the loop on either convergence or iteration cap**
(transfer.go:177-180). If the iteration didn't converge, `IsStable`
makes a stability decision on garbage roots. No signal reaches the
caller.

Plus: **the marginal case (Re(p) = 0) is currently misclassified as
Unstable** by the `>= 0` check at transfer.go:247 — and the API
*can't even express* the four-way split.

Fix: add `Stability` enum + return error on `Poles`:

```go
type Stability int
const (
    StabilityUnknown Stability = iota
    Stable                  // all Re(p) < 0
    MarginallyStable        // some Re(p) == 0, others < 0
    Unstable                // some Re(p) > 0
)

func (tf *TransferFunction) Stability() Stability
func (tf *TransferFunction) Poles() ([]complex128, error)
```

Keep `IsStable() bool` as cheap-cheerful wrapper (caller accepts
silent-failure tradeoff). Three methods, three contracts, each
documented.

---

## 7. `Poles()` has no companion methods

No `Zeros()`, `DCGain()`, `HighFrequencyGain()`, `RolloffSlope()`.
Every site needing both poles+zeros (Bode margins, root-locus,
minimum-phase test) would call them separately. python-control has
all as peer methods; MATLAB CST has `zpk` as a first-class type.
Add peer methods (~50 LOC) **plus** a `PoleZeroData` analysis struct
(`{Poles,Zeros []complex128; DCGain, Gain float64}`) for
cross-language golden vectors.

For `Bode`/`Nyquist`, **CLAUDE.md mandates output buffers** — not
parallel-slice tuple return (forces double-allocation, ugly JSON):
`Bode(sys, ω, outMag, outPhase)` and `Nyquist(sys, ω, out
[]complex128)`. For margins, return a struct (5 floats + bool;
python-control's 5-tuple `(gm, pm, sm, wpcg, wgcg)` is widely
disliked) — see §11 for the `StabilityMargins` shape.

---

## 8. Filter naming: 4 schemes for 4 functions

Reality conflates frequency-domain filtering (`LowPassFilter`,
`HighPassFilter` — actual first-order filters) with sensor fusion
(`ComplementaryFilter` — IMU fusion, not a filter) with slew-rate
limiting (`RateLimiter` — not a filter). When 052-Tier-2 adds
`SecondOrderLowPass`, `Butterworth(order, fc)`, `Notch(ω, Q)`,
`Biquad`, the cluster needs taxonomy. **Modelica.Blocks.Continuous**
is canonical (Z-N / Cohen-Coon / AMIGO auto-tuning expect Modelica
vocabulary): `FirstOrder(k, T)`, `SecondOrder(w, D, k)`,
`Integrator(k)`, `Derivative(k, T)`, `LowpassFilter(order, fc)`
(Butterworth / Bessel / Cheby family), `HighpassFilter`,
`BandpassFilter`, `LimRateLimiter(maxRate)`, `LimPID`. Migration:
rename `LowPassFilter → FirstOrderLowPass`, `HighPassFilter →
FirstOrderHighPass`, `ComplementaryFilter → ComplementaryFusion`,
`RateLimiter → SlewRateLimit`. Keep originals as deprecated aliases.

---

## 9. `TransferFunction.{Num,Den}` exported = no invariants

`Numerator/Denominator []float64` are exported. **No `New(num, den)`
constructor** — Reality uses struct-literal init, no validation runs.
Caller can set `Denominator = nil` between calls; next `Evaluate`
panics. When 053-Tier-1's `dt` lands and `Bode` results need caching,
exported-mutable slice defeats the cache: any external write
invalidates cached poles silently. Slice header is exported, so even
with `New`, the backing array can be mutated. Fix: `New(num, den)
(*TransferFunction, error)` validates and deep-copies; `Num()/Den()`
accessors return defensive copies; make fields private. Also unblocks
`dt float64` private field + `Continuous()/SamplePeriod()` accessors
per 053-Tier-1.

---

## 10. Error policy: one doc-comment fix

Programmer errors (empty/nil/dim) → panic. Numeric out-of-range
(alpha ∉ [0,1], dt ≤ 0) → clamp/no-op, **documented per-function**.
Iterative non-convergence (Durand-Kerner, future Riccati / LQR) →
return error. Zero code, but unblocks 052-Tier-1 (Bode/Nyquist/
Margins) which need to know whether a non-convergent pole-find should
kill the program. Test gap: clamp behavior is untested today (no unit
asserts that `LowPassFilter(0,1,1.5) == LowPassFilter(0,1,1.0)`);
Durand-Kerner divergence has zero golden vectors.

---

## 11. One-PR consolidation

```go
// LTI interface (053-Tier-1 borrow).
type LTI interface {
    Evaluate(s complex128) complex128
    Poles() ([]complex128, error)
    Zeros() ([]complex128, error)
    DCGain() float64
    SamplePeriod() float64                          // 0 = continuous
}

// Stability classification (§6).
type Stability int
const (StabilityUnknown Stability = iota; Stable; MarginallyStable; Unstable)

// Functional-options PID (§5).
type PIDOption func(*PIDController)
func NewPID(kp, ki, kd float64, opts ...PIDOption) *PIDController
func WithLimits(min, max float64) PIDOption
func WithDerivativeOnMeasurement() PIDOption
func WithAntiWindup(mode AntiWindupMode) PIDOption
func (p *PIDController) UpdateError(err, dt float64) float64           // §4

// StatefulFilter interface (§3).
type StatefulFilter interface {
    Update(input, dt float64) float64
    Reset()
}

// Margins struct + frequency-domain ops with output buffers (§7).
type StabilityMargins struct {
    GainMargin, PhaseMargin, DelayMargin float64
    GainCrossover, PhaseCrossover        float64
    Stable                                bool
}
func Margins(sys LTI) StabilityMargins
func Bode(sys LTI, omegas, outMag, outPhase []float64)
func Nyquist(sys LTI, omegas []float64, out []complex128)
```

15 new names + 12 existing = **27** — still smaller than every peer.

---

## 12. Future shapes: StateSpace, c2d

```go
type StateSpace struct {
    A, B, C, D *linalg.Matrix   // A:n×n B:n×m C:p×n D:p×m
    dt         float64          // 0 = continuous
}
func NewStateSpace(A,B,C,D *linalg.Matrix, dt float64) (*StateSpace, error)
func TF2SS(tf *TransferFunction) (*StateSpace, error)
func SS2TF(ss *StateSpace, inputIdx, outputIdx int) (*TransferFunction, error)

// ONE function with method enum — NOT separate Tustin / ZOH /
// BackwardEuler funcs forcing user to know all four. Matches
// python-control c2d signature.
type DiscretizationMethod int
const (Tustin DiscretizationMethod = iota; TustinPrewarp; ZOH; FOH;
       ForwardEuler; BackwardEuler; Impulse)
func C2D(sys LTI, dt float64, method DiscretizationMethod) (LTI, error)
func D2C(sys LTI, method DiscretizationMethod) (LTI, error)

// Per-algorithm form for power users (Slycot-style naming).
func TustinDiscretize(tf *TransferFunction, dt float64) (*DiscreteTransferFunction, error)
func ZOHDiscretize(ss *StateSpace, dt float64) (*StateSpace, error)
```

`C2D(sys, dt, method)` is the canonical entry. Implementation = `tf2ss
→ ZOHDiscretize → ss2tf`. Every control engineer has typed `c2d(sys,
0.01, 'tustin')` and will type `control.C2D(sys, 0.01, control.Tustin)`.

---

## 13. python-control parity (gap is 052's; shape is this report's)

| python-control | reality (today) | proposed |
|---|---|---|
| `tf([1],[1,2,1])` | `&TransferFunction{...}` | `New(num, den)` |
| `ss(A,B,C,D)` | absent | `NewStateSpace(A,B,C,D,0)` |
| `tf2ss(tf)` / `ss2tf(ss)` | absent | `TF2SS` / `SS2TF` |
| `c2d(sys,dt,'tustin')` | absent | `C2D(sys, dt, Tustin)` |
| `bode(sys, w)` | absent | `Bode(sys, w, outMag, outPhase)` |
| `margin(sys)` | absent | `Margins(sys) StabilityMargins` |
| `pole(sys)` / `zero(sys)` | `tf.Poles()` / absent | `sys.Poles()` / `sys.Zeros()` (LTI) |
| `pid(kp,ki,kd)` | `NewPID(kp,ki,kd,min,max)` | `NewPID(kp,ki,kd, WithLimits(...))` |
| `dcgain(sys)` | absent | `sys.DCGain()` |
| `lqr(A,B,Q,R)` | absent | future (052) |

---

## 14. Recommendations, ordered by impact ÷ cost

| # | Change | LOC | Breaking? | Impact |
|---|---|---|---|---|
| 1 | Document error policy (§10) | 0 | no | high |
| 2 | `Stability` enum + method (§6) | 30 | no | high |
| 3 | `Poles() ([], error)` (§6) | 15 | yes | high |
| 4 | `Zeros`, `DCGain`, `RolloffSlope` (§7) | 60 | no | medium |
| 5 | `UpdateError(err, dt)` (§4) | 20 | no | medium |
| 6 | Functional-options `NewPID` (§5) | 80 | yes | high |
| 7 | `LTI` interface (§11) | 30 | no | high |
| 8 | `StatefulFilter` interface (§3) | 15 | no | medium |
| 9 | Modelica filter aliases (§8) | 10 | no | medium |
| 10 | Private TF.Num/Den + `New` (§9) | 50 | yes | high |
| 11 | `C2D(sys,dt,method)` enum-dispatch (§12) | 0 (design) | no | high |
| 12 | Output-buffer `Bode`/`Nyquist` (§7) | 0 (design) | no | high |

**Critical-path PR** (~155 LOC, non-breaking): items 1, 2, 4, 5, 7,
8, 9. Lands API consolidation without invalidating any current
Pistachio/Pulse/Sentinel call site. Items 3, 6, 10 are breaking — one
separate PR with deprecation note.

---

## 15. Out of scope

Numerical correctness (kick / complementary algebra / Durand-Kerner
conditioning / IEEE-754) — 051. Adding LQR / Kalman / MPC / Bode /
Nyquist algorithms themselves — 052. LTI interface / dt discriminator
/ block-algebra as **patterns** — 053. This report owns the **Go
signature shapes** of those patterns. Allocation profile of `Poles`,
matrix-exp speed for `c2d`, `make([]complex128)` per call — 055. 055
will need to evaluate whether §11's `Bode(sys, ω, outMag, outPhase)`
holds at 60 FPS for Pistachio loop-shaping, and whether
`StateSpace.A *linalg.Matrix` forces allocations in the `c2d` hot path.

---

**Files:** `control/pid.go`, `control/transfer.go`, `control/filter.go`,
`control/control_test.go`; sibling reviews 051/052/053 in
`reviews/overnight-400/agents/`.

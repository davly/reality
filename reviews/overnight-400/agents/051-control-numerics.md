# 051 | control-numerics

**Topic:** numerical-correctness audit of `control/` — PID windup, transfer-function pole-zero numerical stability, Bode at ω→0/∞.

**Files audited (all under `C:/limitless/foundation/reality/control/`):**
- `pid.go` (123 LOC) — PIDController struct + NewPID + Update + Reset.
- `transfer.go` (253 LOC) — TransferFunction, Evaluate, Poles, IsStable, Durand-Kerner root finder, internal cos/sin Taylor.
- `filter.go` (117 LOC) — LowPassFilter, HighPassFilter, ComplementaryFilter, RateLimiter.
- `control_test.go` (598 LOC), `control_edge_test.go` (236 LOC) — unit + edge tests.
- `testdata/control/pid_step_response.json` (214 LOC) — single golden-file vector set, PID step response only.

**Net verdict.** PID and TF surface is shallower than the standard control-engineering canon. PID does anti-windup by *back-out integration on saturation* (correct) but is missing four well-known features: derivative kick suppression, derivative filtering, back-calculation tracking, setpoint weighting. Discretization is **forward Euler**, not Tustin (the industry default), and this is undocumented. Transfer function has only continuous-time s-domain — no discretization (ZOH/Tustin), no Bode/Nyquist primitive, no margins, no step response, no LQR, no Kalman. Durand-Kerner pole finder is a single-precision implementation with two bugs (ad-hoc Cauchy-bound, naive deflation-free iteration) that fails to converge on multiple-root and tightly-clustered-root polynomials. No high-order golden-file vectors. List of ranked numerical-correctness findings follows.

---

## §1 PID Controller — `pid.go`

### 1.1 [HIGH] Anti-windup uses *symmetric back-out*; correct in principle but misses 3 standard variants

The implementation (`pid.go:97–110`):
```
output := pTerm + iTerm + dTerm
if output > p.maxOutput {
    if dt > 0 { p.integralSum -= err * dt }   // undo this step's integral contribution
    output = p.maxOutput
} else if output < p.minOutput {
    if dt > 0 { p.integralSum -= err * dt }
    output = p.minOutput
}
```

This is one of the four canonical anti-windup schemes (Astrom & Murray *Feedback Systems* §11.3 names them: clamping, conditional integration, back-calculation, and observer-based). Reality ships only **clamping** (also called "saturation back-out" / "integral clamping"). Properties:

- ✓ **No windup growth while saturated** — integral cannot accumulate past the saturation point in the *direction* of saturation.
- ✗ **Asymmetry under sign flip not handled.** The check is `output > maxOutput` (post-saturation) but `err` could still be the wrong sign; e.g. if the controller is *positively* saturated and the error reverses to negative, the integral *should* be allowed to decrease (this works because `err*dt < 0` so `integralSum -= err*dt` actually adds, which matches the desired direction — but the algorithm's invariant "undo whatever I just added" is non-obvious and hides the correctness argument).
- ✗ **No back-calculation tracking gain** (`Kt` in the standard `du = Kt * (u_sat - u)` formulation). Back-calculation lets the integrator approach the correct value smoothly with time constant `Tt = 1/Kt`, while clamping makes it stop instantly. For systems with slow dominant pole, back-calculation has measurably better recovery from saturation.
- ✗ **Conditional integration** (don't integrate when `|err|` is large *and* would push deeper into saturation) is a separate scheme that's even simpler than clamping and avoids the "undo" pattern.

**Action C-PID-AW-1 (~50 LOC):** add `Kt` field to `PIDController`, default 0 (clamping mode); when `Kt > 0`, replace lines 99–110 with `p.integralSum += Kt * (u_clamped - u_unclamped) * dt`. Document that clamping (Kt=0) is the default and back-calculation (Kt>0) is opt-in. Tunes: `Kt = 1/Ti = Ki/Kp` is a typical starting point.

### 1.2 [HIGH] Derivative kick: Kd term operates on error, not on measurement

`pid.go:91–93`:
```
if dt > 0 {
    dTerm = p.Kd * (err - p.prevError) / dt
}
```

When the **setpoint is stepped** (e.g. operator types in a new target), `err` jumps discontinuously, so `(err - prevError)/dt` blows up. Industrial PID convention (since Astrom 1995, *PID Controllers: Theory, Design, and Tuning*) is **derivative-on-measurement only**: `dTerm = -Kd * (measured - prevMeasured) / dt`. Sign flip because `d(setpoint - measured)/dt = -d(measured)/dt` at constant setpoint, and the discontinuity at setpoint changes is suppressed.

The Reality implementation will produce a pathological "spike" of `Kd * Δsetpoint / dt` whenever the setpoint changes — for `Kd=1.5`, `Δsetpoint=10`, `dt=0.01` (typical 100 Hz loop), the spike is 1500. **Pistachio's camera controller using setpoint changes (cuts) will see exactly this artifact.**

This is sufficiently load-bearing that it's untracked in the test suite — `TestPID_DerivativeDampsOvershoot` (`control_test.go:78`) uses a *constant* setpoint of 1.0 and never steps it, so the kick is invisible.

**Action C-PID-KICK-1 (~10 LOC):** add `prevMeasured float64` to struct; replace `dTerm` computation with `dTerm = -p.Kd * (measured - p.prevMeasured) / dt`; update `Reset()` to clear `prevMeasured`; add golden-file case "setpoint step at t=2s" that demonstrates the difference (without the fix, output spikes; with the fix, smooth).

### 1.3 [HIGH] No derivative filtering — pure differentiator amplifies high-frequency noise

The derivative term is `Kd * Δerror / dt` — a *pure* discrete differentiator with frequency response `|H(jω)| = Kd * ω`, unbounded as ω → ∞. Any measurement noise at the Nyquist frequency is amplified by `Kd / dt` per cycle. Standard practice is the **filtered derivative**:

```
   sKd
H_d(s) = ─────────       (typical N ∈ [8, 20], cutoff at N/Td rad/s)
        1 + sKd/N
```

In discrete-time backward-Euler this is one extra state with one extra multiply. Astrom & Murray §10.5 calls this "essential for any practical PID" — without it, the D term must be set to zero in any system with measurement noise above ~1% of full scale, which means in practice the user can either filter externally (extra burden) or set Kd=0 (which makes the controller a PI, defeating the purpose).

**Action C-PID-DFILT-1 (~20 LOC):** add `dFilterN float64` field (0 disables); when `> 0`, run `dTerm` through `dTerm_f = dTerm_f * (1 - α) + dTerm * α` where `α = N*dt / (Kd + N*dt)`; default N=10 if Kd>0 and user didn't set it; document with cite to Astrom §10.5.

### 1.4 [MEDIUM] Discretization scheme is implicit forward Euler — undocumented and nonstandard

The integral uses `integralSum += err * dt` (forward Euler / left-Riemann sum). The derivative uses `(err - prevError) / dt` (backward Euler difference). The combination is **inconsistent** — integral is forward Euler, derivative is backward Euler.

Industry standard is one of:
- **Backward Euler** throughout (`s ≈ (1 - z⁻¹)/T`): integral is `sum += err_new * dt`, derivative is `(err_new - err_old)/dt`. Stable for any `dt`, no algebraic loop. This is what Reality's *derivative* uses.
- **Tustin / bilinear** (`s ≈ (2/T)(1-z⁻¹)/(1+z⁻¹)`): preserves stability and frequency response down to Nyquist. Requires `(err_new + err_old)/2 * dt` for integral. Recommended by IEEE 1351-2018 PID standard.
- **Forward Euler** (`s ≈ (1 - z⁻¹)/T` shifted): integral is `sum += err_old * dt`. *Not* stable for all `dt` — system can go unstable at large `dt`. This matches Reality's *integral*.

Reality's actual scheme (forward Euler integral + backward Euler derivative) is a frankenstein — it has the stability *flaws* of forward Euler (integral can destabilize at large `dt`) without the *benefits* of consistency. The discretization choice is *undocumented* in the package doc-comment.

**Action C-PID-DISC-1 (~15 LOC):** document the discretization choice in `pid.go` package doc; switch integral to backward Euler (use `err`, the *current* error, after the update — though this is what the code does in practice if you read carefully, since `integralSum` is updated with the *new* `err` not `prevError`). Actually re-reading `pid.go:84–87`: yes, it's backward Euler integral (uses current `err`), so the package is consistently backward Euler throughout. Doc-comment claim. Drop "forward Euler integral" finding above. **Revised:** the implementation IS consistent backward Euler; the issue is purely documentation. Add one sentence: "Discretization: backward Euler (`s ≈ (1 - z⁻¹)/T`). For better high-frequency fidelity, see TODO Tustin variant."

### 1.5 [MEDIUM] No setpoint weighting (`b`, `c` parameters)

Modern PID (Astrom-Hagglund 1995, ISA-S5.1) uses **setpoint weighting**:
```
P = Kp * (b*setpoint - measured)        (b ∈ [0,1], default 1)
D = Kd * (c*setpoint - measured)        (c usually 0 — derivative-on-measurement)
I = Ki * (setpoint - measured)
```
With `b < 1`, the proportional response to a setpoint step is reduced (smoother), while disturbance rejection is unchanged. Setting `c = 0` is exactly the derivative-kick fix from §1.2.

**Action C-PID-SPW-1 (~15 LOC):** add `Pweight, Dweight float64` fields (defaults 1.0, 0.0). Compute `pTerm = Kp * (Pweight*setpoint - measured)`. Replace derivative term per §1.2 with `Kd * (Dweight*setpoint - measured - (Dweight*prevSetpoint - prevMeasured)) / dt`.

### 1.6 [LOW] `dt <= 0` short-circuit silently masks errors

`pid.go:84, 91, 101, 106`: `dt > 0` check guards I and D updates, but no panic, no return error. A caller passing `dt=0` (clock not advanced) gets a P-only result silently, which can mask wall-clock bugs.

**Action C-PID-DT-1 (~5 LOC):** doc explicitly that dt≤0 returns P-only; consider `panic` for `dt < 0` (negative dt is meaningless in causal control), keep `dt == 0` as no-update P-only.

### 1.7 [LOW] No protection against `Kp == 0 && Ki == 0 && Kd == 0` no-op + no Output() inspection

A degenerate controller silently outputs 0; tested at `control_test.go:180–184` (TestPID_ZeroGains). Not a bug per se but ergonomics — no `Output()` method, no way to introspect last `pTerm/iTerm/dTerm` decomposition for debugging or telemetry.

**Action C-PID-INTROSPECT-1 (~20 LOC):** add `LastTerms() (p, i, d float64)` returning the unclamped contributions from the last `Update`. Critical for tuning sessions and Pulse-style monitoring.

---

## §2 TransferFunction — `transfer.go`

### 2.1 [HIGH] Durand-Kerner root finder fails on repeated and clustered roots

`transfer.go:114–183`. Durand-Kerner (Weierstrass) iteration converges quadratically for **simple, well-separated roots**, but:
- **Multiple roots** (e.g. `(s+1)^3`): convergence drops to *linear* and tolerance `1e-12` on the iteration delta is meaningless — final roots can be off by `~ε^(1/m)` where m is multiplicity (so for m=3, ε^(1/3) ≈ 1e-5 even at machine eps).
- **Clustered roots** (e.g. `(s+1)(s+1.001)`): the `denom = ∏(roots[i] - roots[j])` becomes near-zero, the correction `delta = val/denom` blows up, and roots can fly off into Re(s) > 0, falsely flagging a stable system as unstable.
- **High-order polynomials** (degree > 10): combination of the two; Cauchy-bound initial guess `r = max|a_i|` is loose (true bound is `1 + max|a_i|/|a_n|`); 1000 iter cap may be insufficient.

The test suite tests up to degree 3 (`TestTransfer_Poles_ThirdOrder` at `control_test.go:462`) and uses tolerance `1e-6`. **No test for repeated roots, no test for clustered roots, no test for degree ≥ 4.**

**Action C-TF-ROOTS-1 (~80 LOC):** replace Durand-Kerner with **Aberth–Ehrlich** method (a strict improvement on Durand-Kerner: same simultaneous-iteration structure, but uses the *Newton* correction `p(z)/p'(z)` as the numerator instead of just `p(z)`, giving cubic convergence for simple roots and *quadratic* for multiple roots) OR use **balanced companion matrix + QR** (numerically stable, used by NumPy `roots()` and MATLAB `roots()`). Alternative if 80 LOC is too much: ship Durand-Kerner as the easy path but warn in doc and add a `Poles()` returning `(roots []complex128, residual float64)` signature so caller can verify quality.

**Action C-TF-ROOTS-TEST-1 (~30 LOC):** add golden-file vectors at degrees 4, 6, 10 with both well-separated and clustered roots; add explicit test for repeated roots `(s+1)^3`, `(s+1)^2 * (s+2)`.

### 2.2 [HIGH] No conditioning check on companion matrix; no leading-zero stripping

`transfer.go:86–89`: `lead := d[0]`; panic if zero. But what about `[1e-300, 1, 1]`? Leading coefficient near zero implies near-degenerate polynomial — the *actual* degree is lower. Standard fix is to strip leading zeros AND warn. Reality does neither.

Also: `make([]float64, len(d))` allocation on every `Poles()` call — fine for occasional analysis, not fine for hot-path Bode sweep that calls Poles repeatedly.

**Action C-TF-COND-1 (~10 LOC):** strip leading near-zero coefficients (threshold `1e-15 * max|coeff|`); document as "effective degree may differ from len(Denominator)".

### 2.3 [HIGH] No Bode-plot primitive, no margins, no Nyquist

The package documents itself as "classical control theory primitives" (`pid.go:1`), but the toolbox is bare: `Evaluate(s)` gives you `H(jω)` for one frequency, but no `Bode(omegas []float64) (mag, phase []float64)`, no `GainMargin / PhaseMargin`, no `Nyquist` count of encirclements, no `StepResponse`, no `ImpulseResponse`. These are the primary outputs of a control package.

The CLAUDE.md topic prompt explicitly asks: "Bode plot: ω→0 (DC gain) and ω→∞ (high-frequency rolloff) — log scale safety." None exist.

**Numerical concerns when these are added:**
- ω→0: at ω=0, `H(0) = numerator[end] / denominator[end]` directly — no need to call `Evaluate(complex(0,0))` and risk Horner cancellation. Provide a `DCGain()` shortcut that does this exact computation.
- ω→∞: at large ω, `H(jω) ≈ num[0] / (den[0] * (jω)^(m-n))` — high-order Horner accumulates error; provide an asymptotic shortcut for large ω.
- Log-spaced frequency sweep: `omega = 10^linspace(log10(omega_min), log10(omega_max), N)`. Watch for overflow when `omega^k` for large polynomial degree k — at ω = 10^6 with degree 6, `ω^6 = 10^36`, fine; at ω = 10^20 with degree 6, `ω^6 = 10^120` underflows nothing but loses precision in numerator-denominator difference.
- Phase unwrapping: `cmplx.Phase()` returns in (-π, π]; consecutive frequency points need unwrapping to monotone phase. Standard algo: detect jumps > π and add ±2π.
- Gain margin: smallest |1/H(jω)| where ∠H(jω) = -180°; needs root-finding on phase function.
- Phase margin: 180° + ∠H(jω_gc) where |H(jω_gc)| = 1; needs root-finding on log-magnitude.

**Action C-TF-BODE-1 (~80 LOC):** ship `Bode(omegas []float64) (mag, phase []float64)`, `DCGain() float64`, `HighFrequencyRolloff() (slope_dB_per_decade int)`. ω→0 and ω→∞ closed-form shortcuts. Phase unwrap. Golden-file vectors over 5 decades for first/second/fourth-order systems.

**Action C-TF-MARGINS-1 (~50 LOC):** ship `GainMargin() float64`, `PhaseMargin() float64`, `BandwidthCrossover() float64`. Bracket-and-bisect on Bode arrays.

### 2.4 [MEDIUM] Internal `realCos`/`realSin` Taylor-series implementations are inferior to `math.Cos`/`math.Sin`

`transfer.go:191–225`. The package eschews `math.Cos`/`math.Sin` "to keep the dependency list minimal" — but `math` is already imported transitively (`math/cmplx` imports `math`). The Taylor implementation has only 12 terms and the doc-comment says "Reduce to [0, 2π]" via repeated subtraction (`for x >= twoPi { x -= twoPi }`), which:
- Loses precision for large `x` (each subtraction of the `twoPi` literal `6.283185307179586` rounds; for `x = 1e10`, you'd subtract ~1.6e9 times, accumulating ~1e9 ulps of error).
- Is `O(x / 2π)` — for any large argument it's a hot loop.
- For only-12-term Taylor, the error near `x = π` (worst case after reduction) is `x^25/25! ≈ π^25/25! ≈ 2.5e-13` — actually fine for the Durand-Kerner *initial guess*, but the loop above masks this.

This code is only used to lay out initial-guess roots on a circle. The whole sub-block is unjustified — `math.Sin`/`math.Cos` are zero-dependency-cost at this point.

**Action C-TF-TRIG-1 (~20 LOC delete):** remove `realCos`, `realSin`; replace with `math.Cos`, `math.Sin`. Net LOC reduction.

### 2.5 [MEDIUM] No discretization (ZOH, Tustin) — continuous TF cannot be discretized

A control package without a continuous-to-discrete bridge is half-built. Need:
- **Zero-order hold** (`tf.DiscretizeZOH(T) DiscreteTransferFunction`): exact for piecewise-constant inputs (the standard model for digital control). Implementation: `Φ = exp(A*T)`, `Γ = ∫₀^T exp(A*τ) dτ * B` — requires a state-space form; or for SISO `H(s)`, apply Astrom's "exp(A*T) via Padé[5/5]" shortcut.
- **Tustin** (`tf.DiscretizeTustin(T) DiscreteTransferFunction`): substitute `s ← (2/T)(1-z⁻¹)/(1+z⁻¹)`, expand. Closed form on the polynomial coefficients.
- **`DiscreteTransferFunction` type**: separate struct with z-domain coefficients; `Evaluate(z complex128)`, `Poles()` (different stability criterion: `|p| < 1`).

**Action C-TF-DISC-1 (~120 LOC):** new `discrete.go`. Tustin first (algebraic, no eigenvalue computation needed); ZOH later (needs matrix exponential, which Reality doesn't have — it would need linalg dependency or a Padé[6/6] inline routine ~40 LOC).

### 2.6 [LOW] `Evaluate` at exact pole returns `complex128` with `Inf` — but `cmplx.IsInf` handling not documented

`transfer.go:36–48`: division `num / den` where `den = 0+0i` returns `(NaN+NaNi)` per Go spec, not `Inf`. The test `TestTransfer_Evaluate_AtPole` (`control_edge_test.go:154`) asserts `cmplx.IsInf` — this passes only because `1/0 = +Inf` in float64 division for `complex128(re=1, im=0) / complex128(re=0, im=0)` which Go's `complex` division *does* propagate to `(Inf, NaN)` via the textbook formula. Edge case for `(1+0i)/(0+0i)`:
```
(a+bi)/(c+di) = (ac+bd)/(c²+d²) + (bc-ad)/(c²+d²) i
              = (1*0+0*0)/(0+0) + (0*0-1*0)/(0+0) i
              = 0/0 + 0/0 i
              = NaN + NaN i
```
Go's stdlib `complex128` division uses Smith's algorithm which handles this differently. **Empirical:** `complex(1,0)/complex(0,0) → (+Inf+NaNi)` in current Go (1.21+). `cmplx.IsInf` returns true if *any* part is Inf, so the test passes. But documentation should state this explicitly.

**Action C-TF-POLE-EVAL-1 (~5 LOC):** doc-comment add: "If `s` is exactly a pole of D, returns a complex value with at least one Inf component (test with `cmplx.IsInf`). Behavior near a pole is well-defined: |H(s)| → ∞ as s → pole."

---

## §3 Filters — `filter.go`

### 3.1 [MEDIUM] `LowPassFilter` and `HighPassFilter` use `α` parameter not cutoff frequency

`filter.go:15`, `filter.go:38`: API takes raw `alpha ∈ [0,1]`, not a cutoff `fc` or `RC`. Engineering practice expects:
```
α = dt / (RC + dt)         where RC = 1/(2π fc)
```
The user must derive α externally. This is consistent with Pistachio's "camera damping" use case (where α is a tuning knob), but inconsistent with the signal/ package which DOES expose Hz cutoffs.

**Action C-FILT-CUTOFF-1 (~20 LOC):** add `LowPassFilterFc(prev, current, fc, dt) float64` that converts cutoff to α internally.

### 3.2 [LOW] `HighPassFilter` formula is non-standard

`filter.go:38–46`: `out = α * (prevFiltered + current - prev)`. The textbook discrete-time high-pass is:
```
y_n = α * (y_{n-1} + x_n - x_{n-1})         where α = RC / (RC + dt)
```
where `α → 1` means *long* time constant (preserves low frequency only above cutoff — i.e. high-pass), and `α → 0` means *short* time constant. Reality's parameter direction matches but the doc-comment says "alpha close to 1 → more high-frequency content preserved" which is misleading: a highpass with α near 1 has a *low cutoff*, so it preserves *more* frequencies — including most of the band. The doc is approximately correct but ambiguous.

**Action C-FILT-DOC-1 (~5 LOC):** clarify doc as "α = RC/(RC+dt); cutoff ω_c = 1/RC; small α = high cutoff (only fast transients pass), large α = low cutoff (most signal passes)".

### 3.3 [LOW] `ComplementaryFilter` collapses to `accel + α*gyro*dt` regardless of α

`filter.go:84`: `α*(accel + gyro*dt) + (1-α)*accel`. Algebraic simplification:
```
= α*accel + α*gyro*dt + accel - α*accel
= accel + α*gyro*dt
```
The `α` weighting on the accel-only branch is completely cancelled by the same `α` weighting on the (accel + gyro·dt) branch. The function effectively does `accel + α*(gyro*dt)`, which is **not a complementary filter at all** — the standard formula is:
```
θ_n = α * (θ_{n-1} + gyro * dt) + (1-α) * accel_angle
```
Note: it uses `θ_{n-1}` (the *previous filtered angle*) not `accel`. Reality's signature lacks the previous filtered angle. With `accel` taking that slot, the recursion is broken — there's no integration of gyro across multiple steps.

**This is a real bug.** A user calling `ComplementaryFilter(accel, gyro, 0.98, dt)` gets back `accel + 0.0196*gyro*dt`, which is essentially just the accelerometer reading with a tiny gyro nudge — not a fused estimate. Pistachio's IMU sensor fusion is broken if it depends on this.

**Action C-FILT-COMP-1 (~15 LOC, BREAKING):** redefine signature to `ComplementaryFilter(prevAngle, accelAngle, gyroRate, alpha, dt) float64` returning `α*(prevAngle + gyroRate*dt) + (1-α)*accelAngle`. Update tests; current tests at `control_test.go:269–295` and `control_edge_test.go:106–122` are also wrong by the same mathematical identity (they test the broken formula). Document migration in CHANGELOG.

### 3.4 [LOW] `RateLimiter` boundary check `delta > maxDelta` is asymmetric for negative-direction input

`filter.go:103–116`: code is correct (`delta = target - current`; `if delta > maxDelta: current + maxDelta; if delta < -maxDelta: current - maxDelta`). No issue. **Withdrawn.**

---

## §4 Test infrastructure — coverage gaps

### 4.1 [HIGH] Single golden-file (`pid_step_response.json`) covers only PID; nothing else

`testdata/control/` has exactly one JSON. Per CLAUDE.md design rule "every function has golden-file test vectors," missing coverage:
- `LowPassFilter` — 0 vectors.
- `HighPassFilter` — 0 vectors.
- `ComplementaryFilter` — 0 vectors (and would expose the bug in §3.3).
- `RateLimiter` — 0 vectors.
- `TransferFunction.Evaluate` — 0 vectors (exists at multiple s).
- `TransferFunction.Poles` — 0 vectors (especially for degree > 3).
- `TransferFunction.IsStable` — 0 vectors.

The 22-package codebase has 1,965 tests but `control/` carries the burden almost entirely in non-golden Go-only unit tests, breaking the cross-language validation property.

**Action C-TEST-GOLDEN-1 (~250 LOC of JSON):** ship `lowpass.json`, `highpass.json`, `complementary.json`, `rate_limiter.json`, `tf_evaluate.json`, `tf_poles.json`, `tf_stable.json`. Each ≥20 vectors per CLAUDE.md target.

### 4.2 [MEDIUM] No edge cases for IEEE 754 mandatory inputs

CLAUDE.md "IEEE 754 edge cases mandatory: +Inf, -Inf, NaN, -0.0, subnormals". Current tests don't probe:
- PID with `setpoint = NaN` or `Inf` → output should be NaN propagating, but does anti-windup code path handle `NaN > maxOutput` correctly? (NaN comparisons all return false, so neither clamp branch triggers, and `output = pTerm + iTerm + dTerm = NaN` is returned — actually correct.)
- TransferFunction with `Inf` coefficient → polynomial evaluation overflows.
- LowPass with `prev = -0.0`, `α = 0.5` → result should be `-0.0` (signed-zero correct propagation).
- RateLimiter with subnormal `dt = 1e-310` → `maxDelta = maxRate * 1e-310` underflows to 0; behavior is "limited to 0 change" which is correct but undocumented.

**Action C-TEST-IEEE-1 (~50 LOC):** add edge tests; document expected NaN/Inf propagation in package doc.

### 4.3 [MEDIUM] PID step-response test feeds output as `measured += output * dt` — that's a pure integrator plant, not real test plant

`control_test.go:31–35`. The plant is `dx/dt = u`, the simplest possible. Real test should also include first-order plant `dx/dt = -ax + bu` and second-order plant. The `TestPID_DerivativeDampsOvershoot` *does* use a second-order plant (`control_test.go:78–115`) but as a non-golden Go-only test.

**Action C-TEST-PLANT-1 (~30 LOC):** add golden-file `pid_first_order_plant.json` and `pid_second_order_plant.json` so cross-language tests cover realistic dynamics.

---

## §5 Missing primitives (numerics-relevant)

Not strictly numerical-correctness audit but called out by topic prompt:
- **LQR**: Riccati ARE solver. No primitives. Schur or doubling needed; doubling iter is `O(n³)` per step, ~30 steps to converge. ~150 LOC + linalg eigenvalue dependency.
- **Kalman filter**: standard form is *unstable* in finite precision (P matrix loses positive-definiteness over many updates). Need *square-root* form (Bierman-Thornton UDU' or Carlson) — ~200 LOC.
- **State-space `(A, B, C, D)` representation**: precondition for everything above. ~80 LOC.
- **Model conversion**: tf2ss, ss2tf, modal canonical form. ~120 LOC.
- **Step/impulse response**: direct ODE solve via `chaos.RK4` (which Reality already has) or via convolution with discretized impulse. ~50 LOC.

These are all outside the topic's "audit existing for numerical correctness" — flagged as missing but not as bugs.

---

## §6 Highest-leverage bundle (numerical-correctness only)

If only one PR could ship from this audit, it should be:

| Action | LOC | Impact |
|---|---:|---|
| C-PID-KICK-1 (derivative-on-measurement) | 10 | fixes silent setpoint-step bug |
| C-PID-DFILT-1 (derivative filtering) | 20 | makes Kd usable in noise |
| C-FILT-COMP-1 (correct complementary filter) | 15 | fixes silent IMU-fusion bug |
| C-TF-ROOTS-1 (Aberth or companion-QR) | 80 | fixes high-order root finding |
| C-TF-BODE-1 (Bode + DC gain + rolloff) | 80 | unblocks frequency-domain analysis |
| C-TEST-GOLDEN-1 (per-function vectors) | 250 | cross-language validation |
| **Total** | **455 LOC** | Two real bugs fixed, two missing primitives, golden-file coverage |

Of these, **C-PID-KICK-1** and **C-FILT-COMP-1** are the only ones that fix *active correctness bugs*; the rest are missing-feature or precision-improvement.

---

## §7 Disjoint-check appendix

Adjacent control/ slots (numbers from MASTER_PLAN context; not enumerated here):
- 052 (control-missing): owns missing-primitive surface (LQR, Kalman, state-space, response) — §5 above is informative pointer, not territory claim.
- 053 (control-sota): owns library-comparison (MATLAB Control Systems Toolbox, python-control, slycot, harold) — not addressed here.
- 054 (control-api): owns naming/ergonomics (e.g. `Output()` introspection, `b/c` parameter names) — §1.7 brushes against this; defer to 054 for naming choices, this report only flags the missing capability.
- 055 (control-perf): owns allocation/inlining (e.g. `make([]float64, len(d))` in Poles per call) — §2.2 touched this; defer per-call alloc analysis to 055.

This report covers **numerical correctness** only: anti-windup correctness, derivative kick, derivative filtering, discretization scheme documentation, root-finder convergence, condition number handling, log-scale safety at ω→0/∞, signed-zero/NaN/Inf propagation, complementary-filter algebraic identity check, golden-file coverage gaps with numerical implications.

Report at `agents/051-control-numerics.md`, ~370 lines.

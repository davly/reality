# 319 — dive-pll (PLL: type-I/II/III, Costas, gear-shifted, NCO audit)

## Headline
Reality v0.10.0 ships **zero phase-locked-loop machinery** — no `PLL`, `Costas`, `VCO`, `NCO`, `LoopFilter`, `PhaseDetector`, `GearShift*`, `CarrierRecovery`, `FrequencyEstimator` anywhere in the tree (grep returns 0 production hits); the closest existing primitive is `control/pid.go:36` `PIDController`, which is structurally identical to a type-II PI loop filter but is exposed only as a clamped time-domain controller, not as a phase-domain PLL. Day-1 PR: `signal/pll.go` shipping T0 generic framework + T1 type-I + T2 type-II PI ≈ 320 LOC + 30 golden vectors; T3 type-III, T4 Costas (BPSK/QPSK), T5 gear-shifted, T6 free-standing NCO, T7 FFT-bootstrapped carrier recovery deferred to follow-on PRs. **Critical synergy:** PLL is a special case of the steady-state Kalman filter (slot 308) under specific noise models, opening a strong R-MUTUAL-CROSS-VALIDATION 3/3 pin (PLL gains ≡ DARE-derived steady-state Kalman gain on PLL state model).

## Findings

### What exists in reality (full inventory of PLL-adjacent code)
- `control/pid.go:36` — `PIDController{Kp,Ki,Kd,minOutput,maxOutput,integralSum,prevError}`. **Structurally a PI+D loop filter** with anti-windup clamp; the I-term `integralSum += err*dt` (line 85) is exactly the integrator a type-II PLL needs. But the API is `Update(setpoint, measured, dt)` — measurement domain, not phase domain. No phase-wrap handling, no NCO coupling, no PD-output linearisation.
- `control/filter.go:15` — `LowPassFilter(prev, current, alpha)`. 1-pole IIR; could serve as a type-I loop filter, but no helper for `α ↔ ω_n / ζ` (PLL natural-frequency / damping) parameterisation.
- `control/filter.go:38` — `HighPassFilter(prevFiltered, prev, current, alpha)`. Irrelevant to PLL.
- `control/filter.go:74` — `ComplementaryFilter`. Sensor fusion; sometimes confused with PLL but distinct (no closed-loop frequency tracking).
- `control/transfer.go:21` — `TransferFunction{Numerator,Denominator}` continuous-time only; `Poles`/`Zeros`/`IsStable` (lines 73, 239) help analyse PLL closed-loop transfer, but no `s→z` mapping (slot 315 gap), so cannot synthesise discrete-time PLL coefficients via bilinear.
- `signal/fft.go:49,101` — `FFT`/`IFFT`. Carrier-recovery T7 needs this for an initial frequency estimate before PLL locks (Rife-Boorstyn 1974 quadratic-fit on `|FFT|²`).
- `signal/window.go` — Hann/Hamming/Blackman. Useful for windowed FFT in T7, but Kaiser missing (slot 317).
- `audio/pitch/yin.go`, `audio/pitch/mpm.go`, `audio/pitch/autocorrelation.go`, `audio/pitch/subharmonic_summation.go` — **non-PLL pitch trackers**. Block-based ACF / cumulative-mean-normalised-difference. Cannot follow continuous pitch evolution between blocks; a PLL-based vibrato/glissando tracker would complement these.
- `audio/beat/beat.go`, `audio/tempo/tempo.go` — DP-based beat tracker (Ellis 2007) and ACF-based tempo estimator. Phase-locked beat tracking (Hainsworth-Macleod 2003 Bayesian / Davies-Plumbley 2007 PLL-style) is the natural extension and is not present.
- `audio/onset/superflux.go`, `audio/onset/spectral_flux.go`, `audio/onset/complex_domain.go`, `audio/onset/energy.go`, `audio/onset/peak_picking.go` — onset detection feeds beat/tempo; orthogonal to PLL but a PLL-based "tactus tracker" would consume these.
- `chaos/*` — Lorenz, Van der Pol; Van der Pol is a **self-sustained oscillator** that resembles a VCO under forcing but is not exposed as a controllable NCO.
- `prob/timeseries.go` — Levinson-Durbin AR; AR(2) on noisy sinusoid gives the Pisarenko frequency estimate (a one-shot competitor to PLL). Worth flagging as cross-check.

### What is missing (the actual gap)
Authoritative grep across the entire repo for `PLL|PhaseLockedLoop|Costas|\bVCO\b|\bNCO\b|LoopFilter|PhaseDetector|GearShift|CarrierRecovery|FrequencyEstimator` (case-insensitive) returns **0 hits in production code**. Only matches are in `reviews/overnight-400/agents/133-signal-sota.md:133-135` ("steal SDR algorithm library: PLL, AGC, Costas loop") and the `MASTER_PLAN.md:339` slot-319 line itself. Specifically absent:

1. **Generic PLL framework** — interfaces for `PhaseDetector`, `LoopFilter`, `NCO`, with a `PLL.Update(input float64) (output, phaseEst, freqEst float64)` step function. The framework is what lets type-I/II/III/Costas all share golden-file infrastructure.
2. **Phase detector primitives:**
   - **Multiplier (analog Costas)** — `pd = sin(θ_in − θ_vco)` linearised to `Δθ` near lock; outputs `1/2 sin(2Δθ)` (factor-of-two ambiguity for BPSK).
   - **XOR (digital)** — square-wave inputs, range `±π/2`, gain `2/π`.
   - **JK flip-flop / phase-frequency detector (PFD)** — range `±2π`, used in frequency synthesisers, three-state (UP/DOWN/IDLE).
   - **Hogge detector** — for clock-data recovery (CDR), retimes data on its own clock.
   - **Costas demod** — `pd = I·Q` for BPSK, `pd = sgn(I)·Q − sgn(Q)·I` for QPSK (Polyphase Costas, Lindsey-Simon 1973).
3. **Loop filter (the type classification axis):**
   - **Type-I** = pure gain `K_p` after PD → 1st-order closed loop. Zero static phase error for **constant phase** input only; **constant lag** for ramp input (constant frequency offset). ~40 LOC.
   - **Type-II** = PI filter `K_p + K_i/s` after PD → 2nd-order closed loop, two integrators in the loop (one is the VCO, one is the loop filter). Zero static error for **ramp** input (constant frequency offset). The standard for satellite, GPS L1, and digital comms. Parameterisation `(ω_n, ζ)`: `K_p = 2ζω_n / K_VCO`, `K_i = ω_n² / K_VCO`. Gardner 1980 §2.
   - **Type-III** = PII filter (PI plus extra integrator) → 3rd-order closed loop. Zero error for **parabolic** input (constant frequency-rate, i.e. Doppler chirp). Tracks accelerating satellites (LEO Doppler). Less common; conditionally stable. Stephens 2002 §6.4.
4. **NCO / VCO** — phase accumulator `θ[n+1] = θ[n] + ω[n]·T_s`, modulo 2π. Quadrature output `(cos θ, sin θ)`. **Phase-truncation spurs** are the canonical numerical trap (Tierney-Rader-Gold 1971). Floating-point NCO using `math.Mod` is fine for reality precision; `math.Sincos` (Go) gives the quadrature pair. ~80 LOC.
5. **Costas loop (BPSK/QPSK)** — coherent demodulator that resolves carrier phase without a transmitted pilot tone. Two-arm structure: I = LPF(input·cos(θ_vco)), Q = LPF(input·sin(θ_vco)), error = I·Q (BPSK) or hard-decision-driven (QPSK). The **180° phase ambiguity** must be resolved by differential coding or a pilot symbol. Costas 1956 §III. ~150 LOC.
6. **Gear-shifted PLL (variable-bandwidth)** — state machine that runs **wide-bandwidth (large ω_n)** during acquisition for fast pull-in, then **narrow-bandwidth** in tracking for low jitter. Switch is triggered by lock detector (ratio of post-LPF I-energy to Q-energy, or low-passed |error|). Mengali-D'Andrea 1997 §4.5. ~150 LOC including state machine.
7. **Lock detector** — `ρ = ⟨I²⟩ / (⟨I²⟩ + ⟨Q²⟩)` or `⟨cos(2Δθ)⟩` (low-passed). Maps to a continuous "lock confidence" used by gear-shift and as a downstream signal-quality indicator.
8. **FFT-bootstrap carrier recovery** — coarse estimate of carrier frequency by FFT of input + interpolation (Quinn-Fernandes 1991, or Rife-Boorstyn 1974 with quadratic fit on log-magnitude). Initialises NCO close enough that PLL pull-in is guaranteed. ~80 LOC. Avoids the **acquisition cliff** where pull-in time scales as `(Δf / ω_n)²`.
9. **Pull-in / pull-out range, hold range, lock range** — analytical formulas (Gardner 1980 Table 4-1) gating expected behaviour. Useful as `// Doc:` comments and for golden-file regression: feed `Δf > 2ζω_n` and assert `notLocked`.
10. **All-digital PLL (ADPLL)** — Staszewski-Balsara 2006: time-to-digital converter front-end, digital loop filter, digitally controlled oscillator. Niche for reality; defer indefinitely.
11. **Fractional-N PLL** — sigma-delta modulator drives integer divider; out of scope (frequency synthesiser, not signal-processing PLL).

### Cross-package fanout (what this gap blocks)
- **Audio beat tracking** — `audio/beat/beat.go` runs offline DP (Ellis 2007). A streaming PLL-based "tactus" tracker (Davies-Plumbley 2007; Cemgil-Kappen 2003 Bayesian PLL) is the canonical real-time alternative. Phase-locks to onset envelope, outputs both beat times *and* tempo.
- **Audio pitch tracking** — `audio/pitch/yin.go` is block-based; vibrato / portamento need continuous tracking. PLL with frequency-modulated NCO (or coupled-PLL bank for harmonics) gives 0-latency pitch follower.
- **Music synchronisation** — DAW-style sync to live drums; tap-tempo lock; karaoke alignment.
- **Pistachio audio** — analysis-resynthesis pipelines need carrier estimation for FM-synth-like effects.
- **SDR / digital comms (consumer apps via aicore)** — BPSK/QPSK demod **requires** Costas. Without `signal/pll.go`, no path.
- **Power-electronics / grid-tie** — grid-frequency synchronisation uses 3-phase PLL (SRF-PLL, Karimi-Ghartemani 2014). Out of scope today, but T2 is the building block.
- **Atomic clock disciplining** — GPS-disciplined oscillators; type-II PLL with very long time-constant. Same primitive.
- **Pulse-driven control** (`Pulse` per CLAUDE.md consumer list) — feedback loops sometimes need phase-domain stability; PLL-style integral-only loop is the right primitive.
- **Slot 156 (synergy-topology-prob)** and **slot 165 (synergy-sequence-prob)** — neither covers PLL but both touch oscillation-tracking; PLL would slot under a future `synergy-signal-control` slot.
- **Slot 314 (AHRS)** — magnetometer heading is implicitly phase-locked to magnetic north; not a comms PLL but Mahony's complementary filter is conceptually adjacent.
- **Slot 308 (Kalman)** — see §"Kalman ≡ PLL pin" below.
- **Slot 315 (IIR design)** — once bilinear lands, synthesise the loop filter as a discrete-time `TransferFunction` and validate via pole locations.

### Numerical / precision notes
- **Phase wrap**: must keep `θ` bounded near 0 to avoid `float64` precision loss. `math.Mod(θ, 2π)` after every NCO step. Better: keep `θ` in `[-π, π]` and use `math.Atan2(sinθ, cosθ)` reconstruction occasionally.
- **Loop-filter integrator drift**: same anti-windup story as `control/pid.go:99`. Without clamp, frequency-offset transients can saturate the integrator. Type-II PLLs in literature usually do *not* clamp because the phase error self-limits via the PD nonlinearity (`sin(Δθ)` saturates at ±1), but for QPSK Costas the hard-decision PD has unbounded output; needs clamp.
- **PD nonlinearity**: linear analysis (Gardner) assumes `sin(Δθ) ≈ Δθ`. Pull-in dynamics violate this; the **Adler equation** captures cycle-slipping. Golden vectors should include cycle-slip cases.
- **Damping ratio**: `ζ=1/√2` (Butterworth) is the textbook sweet spot — fastest 5% settle, mild overshoot. Ship as default.
- **Sample-rate normalisation**: all gains parameterised in `f_s`-normalised units (`ω_n·T_s`); user passes natural frequency in Hz and sample rate, library does the conversion. Avoids the easy bug of forgetting `T_s` in `K_i = ω_n² T_s² / K_VCO`.
- **Costas branch ambiguity**: BPSK 180° → cos² PD has period π, gives zero error at both `Δθ=0` and `Δθ=π`. QPSK fourth-power PD has period π/2. Differential decoding upstream resolves.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities (saturated)

Six independent pins identified; (a)+(b)+(d) saturate the 3/3 contract.

| # | Path A | Path B | Pin tolerance | LOC |
|---|---|---|---|---|
| (a) | type-II PLL closed-loop transfer (analytical: `H(s) = (2ζω_n s + ω_n²) / (s² + 2ζω_n s + ω_n²)`) sampled at high `f_s` | discrete-time PLL impulse response → DFT → magnitude | 1e-3 (frequency-bin) | ~50 |
| (b) | gear-shifted PLL acquisition time on `Δf=10·ω_n` step | Gardner 1980 Table 4-1 pull-in formula `T_p ≈ (Δf)² / (ζ ω_n³)` | 20% (regression — analytical formula is asymptotic) | ~40 |
| (c) | steady-state phase jitter `σ_θ²` for AWGN input at SNR `ρ` | Viterbi 1966 formula `σ_θ² = 1/(2ρ B_L T)` where `B_L = (ω_n/2)(ζ + 1/(4ζ))` | 30% (Monte-Carlo over 10⁴ trials) | ~80 |
| (d) | Costas BPSK demod recovers transmitted bits at SNR=10 dB | direct phase-detector + π-disambiguator (sign of correlation against known preamble) | exact bit match (≤1 cycle slip in 10⁴ symbols) | ~60 |
| (e) | type-II PLL gains `(K_p, K_i)` | steady-state Kalman gain of LTI model `[θ; ω]_{k+1} = [[1, T];[0,1]]·[θ;ω] + [w_θ; w_ω]; y = θ + v` (DARE solver from slot 308) | 1e-9 element-wise | ~40 |
| (f) | NCO output `cos(ωt)` over 10⁶ samples | direct `math.Cos(ω·k·T_s)` with `k` accumulated as int64 | 1e-12 (no phase drift) | ~20 |

**Pin (e) is the headline result.** PLL-as-Kalman is well known (Patapoutian 1999, "On Phase-Locked Loops and Kalman Filters"; Driessen 1994) but the equivalence is rarely *enforced* by code. Reality can be the first library to ship a regression test that fails the build if PLL gains drift out of Kalman-equivalent range. This is exactly the cross-language fixed-point reality is built for.

### Kalman ≡ PLL synergy (must-call-out)
Under the linear-Gaussian state-space model
```
x_{k+1} = [θ_{k+1}; ω_{k+1}] = [[1, T]; [0, 1]] x_k + w_k,    w_k ~ N(0, Q)
y_k     = θ_k + v_k,                                          v_k ~ N(0, R)
```
the **steady-state** Kalman gain `K_∞ = [K_θ; K_ω]` is precisely the PI loop-filter gain pair of a type-II PLL operating on `(y_k − θ̂_k)` (with appropriate VCO discretisation). Solving DARE gives:
- `K_θ = (ω_n²·T² + 2ζω_n·T) / (1 + ...)` — depends on PSD ratio `Q/R`.
- The **bandwidth-noise tradeoff** in PLL design is *exactly* the `Q/R` choice in Kalman: large Q = large bandwidth = fast tracking, large noise; small Q = narrow bandwidth = slow but clean.

This means once slot 308's DARE solver lands, the **default** type-II gains in `signal/pll.go` should be derived from a single `(σ_process², σ_measurement²)` pair, with `(ω_n, ζ)` as a derived quantity. That is *strictly more principled* than the Gardner-textbook (ω_n, ζ) entry point and should be the recommended API for v0.12.0.

### Lock-detection / acquisition state-machine (T5 detail)
Gear-shift FSM (Mengali-D'Andrea §4.5):
```
ACQ_WIDE      (ω_n_wide,   ζ=1.0)   — pull-in
  |
  | lock_metric > 0.7 for K_dwell_acq samples
  v
TRACK_NARROW  (ω_n_narrow, ζ=0.7)   — clean tracking
  |
  | lock_metric < 0.3 for K_dwell_loss samples
  v
ACQ_WIDE
```
Loop-bandwidth ratio `ω_n_wide / ω_n_narrow` typically 10:1 to 100:1. Critical: when narrowing, **rescale the integrator state** so the loop-filter output is continuous — otherwise a transient kicks the loop out of lock. Gardner 1980 §8.3.

## Concrete recommendations

### Day-1 PR — `signal/pll.go` (T0 + T1 + T2, ~320 LOC + 30 golden vectors)

**Public API:**
```go
// Package signal/pll — phase-locked loops.
// Generic framework + type-I and type-II implementations.

// PhaseDetector returns scalar phase error in radians.
type PhaseDetector interface {
    Detect(input float64, ncoCos, ncoSin float64) float64
}

// LoopFilter maps phase error to NCO frequency correction.
type LoopFilter interface {
    Update(phaseError, dt float64) float64
    Reset()
}

// NCO is a numerically-controlled oscillator with quadrature output.
type NCO struct {
    Phase      float64 // radians, wrapped to [-π, π]
    BaseFreqHz float64
    SampleHz   float64
}
func (n *NCO) Step(freqCorrectionHz float64) (cos, sin float64)
func (n *NCO) Reset()

// PLL combines a PhaseDetector, LoopFilter, and NCO.
type PLL struct {
    PD  PhaseDetector
    LF  LoopFilter
    NCO *NCO
}
func (p *PLL) Update(input float64) (output, phaseEst, freqEstHz float64)

// MultiplierPD: pd = input * sin(θ_vco) (analog Costas-style).
type MultiplierPD struct{}

// TypeIFilter: scalar gain K_p.
type TypeIFilter struct{ Kp float64 }

// TypeIIFilter: PI filter with parameters (ω_n, ζ) and VCO gain K_vco.
type TypeIIFilter struct {
    Kp, Ki      float64
    integralSum float64
}
// NewTypeIIFromBandwidth returns a critically-near-damped (ζ=1/√2) PI filter
// with closed-loop natural frequency ω_n = 2π·bandwidthHz.
func NewTypeIIFromBandwidth(bandwidthHz, sampleHz float64) *TypeIIFilter

// NewTypeIPLL: turnkey type-I PLL with multiplier PD.
func NewTypeIPLL(carrierHz, sampleHz, kp float64) *PLL

// NewTypeIIPLL: turnkey type-II PLL.
func NewTypeIIPLL(carrierHz, sampleHz, bandwidthHz, dampingRatio float64) *PLL
```

**Cross-validation pins (write all from §"R-MUTUAL-CROSS-VALIDATION"):**
- `TestNCONoPhaseDrift` — pin (f), 1M samples, ≤1e-12.
- `TestTypeIIClosedLoopMatchesAnalytical` — pin (a), magnitude response within 1e-3 over 10 frequency bins.
- `TestTypeIIPullInTimeMatchesGardner` — pin (b), regression vs Gardner table.
- `TestTypeIITracksRampZeroError` — feed `θ(t) = ω₀t + αt`, assert `‖phaseError‖ → 0` (defining property of type-II).
- `TestTypeITracksRampWithLag` — same input, assert constant nonzero `phaseError` (defining negative property of type-I).
- `TestTypeIIEqualsKalmanSteadyState` — pin (e), gated on slot 308 DARE solver landing; if absent, `t.Skip("requires control.SolveDARE")` with a `// TODO(post-308):` marker.

**Golden file** (30 vectors): step input `Δθ = π/4`, frequency step `Δf = ω_n/4`, frequency ramp `α = ω_n²/16`, phase noise σ ∈ {0.01, 0.1, 1.0} rad, AWGN at SNR ∈ {0, 10, 20} dB, ω_n / f_s ratios ∈ {0.001, 0.01, 0.1}, ζ ∈ {0.5, 0.707, 1.0}, edge cases (zero input, DC input, saturating step `Δf > 10·ω_n` → cycle slips).

### Day-2 PR — `signal/pll_costas.go` + `signal/pll_typeIII.go` (~230 LOC)

T3 type-III filter (`PII = K_p + K_i/s + K_ii/s²`, ~80 LOC) and T4 BPSK Costas (~150 LOC). Costas comes with the π-disambiguator helper (correlate against pilot bits) and a QPSK variant (sign-of-decision PD, Lindsey-Simon 1973). Add pin (d) (Costas ≡ direct-PD + π-resolver) and (c) (jitter ≡ Viterbi formula).

### Day-3 PR — `signal/pll_gearshift.go` (~150 LOC)

T5 gear-shifted FSM. Two sub-PLLs (wide / narrow), shared NCO, lock-metric estimator. Critical bug pin: **integrator continuity across gear shift** — assert no transient phase glitch > 0.01 rad on shift event. Cite Mengali-D'Andrea §4.5.

### Day-4 PR — `signal/pll_carrier_recovery.go` (~80 LOC)

T7 FFT-bootstrap. Windowed FFT of input → pick max-magnitude bin → quadratic interpolation of log-magnitudes (Quinn-Fernandes 1991) → seed `NCO.BaseFreqHz`. Then run Costas. Avoids the `Δf/ω_n > 1` acquisition cliff; eliminates the long pull-in transient that motivated gear-shift in the first place. **For wideband applications, T7 obsoletes T5.** Document this tradeoff: T5 for narrowband (no `f_s/N` resolution to bootstrap from), T7 for wideband.

### Day-5 PR — `audio/pitch/pll_tracker.go` + `audio/beat/pll_tactus.go` (~300 LOC, gated on Day-1)

Consumer wiring. PLL-pitch using Costas-on-fundamental (Hermes 1988). PLL-beat using Davies-Plumbley 2007 (onset envelope drives a type-II PLL whose NCO is the tactus pulse). Lifts beat tracking from offline-DP to streaming.

### Cross-link recommendations

- **Slot 308 (Kalman)** — *must* coordinate API. Once `control.SolveDARE` lands, expose `signal/pll.NewTypeIIFromKalman(σ_process, σ_meas)` constructor that internally calls DARE; this becomes the **recommended** entry point for v0.12.0 onward.
- **Slot 314 (AHRS)** — Mahony complementary filter has the same PI-on-error structure; document the analogy in `pll.go`'s package comment.
- **Slot 315 (IIR design)** — once bilinear lands, ship a `signal/pll.PLLAsTransferFunctionZ()` helper for closed-loop pole/zero analysis.
- **Slot 165 (synergy-sequence-prob)** — already DONE; PLL slots into a future `synergy-signal-control` synthesis (recommend new slot in next overnight wave).
- **Slot 156 (synergy-topology-prob)** — already DONE; orthogonal.
- **Slot 133 (signal-sota)** — already flagged PLL/AGC/Costas; this review supplies the actionable spec.

### Out-of-scope / explicit non-goals
- ADPLL (Staszewski-Balsara) — float64 reality doesn't need TDC abstractions.
- Fractional-N synthesiser PLL — frequency-synthesis-only, not signal-processing.
- Sub-sample-period delay-locked loops (DLL) — sibling primitive (timing recovery, not phase recovery); separate slot.
- Polyphase Costas for higher-order modulations (M-PSK, M ≥ 8) — defer to a focused QAM/PSK demod slot.
- Hogge / Alexander phase detectors for clock-data recovery (CDR) — separate "data-aided synchroniser" slot.

### Cheapest day-1 PR summary
T0 (framework) + T1 (type-I) + T2 (type-II PI) = **~320 LOC + 30 golden vectors + 6 cross-validation tests + 1 Kalman-equivalence test (deferred until slot 308 DARE)**. Ships type-II — the standard for all comms/timing/audio applications — and the framework that all subsequent loops (Costas, gear-shift, type-III, carrier recovery) plug into. Critically: T0 forces the `PhaseDetector` / `LoopFilter` / `NCO` interface decisions early so they don't have to be rewritten when T4-T7 land.

## Sources

### Reality codebase (file:line)
- `control/pid.go:36` — `PIDController` (PI+D structure, anti-windup at lines 99-110; reusable conceptually for type-II loop filter).
- `control/filter.go:15` — `LowPassFilter` 1-pole IIR (could serve as type-I loop filter).
- `control/transfer.go:21` — `TransferFunction` continuous-time (no `s→z` for closed-loop pole analysis).
- `signal/fft.go:49,101,167` — `FFT/IFFT/FFTFrequencies` for T7 carrier recovery.
- `signal/window.go:15,44,76` — Hann/Hamming/Blackman (for windowed FFT in T7).
- `signal/filter.go:19,54,97,130` — Convolve/MovingAverage/EMA/Median (loop filter atoms).
- `audio/pitch/yin.go`, `audio/pitch/mpm.go`, `audio/pitch/autocorrelation.go`, `audio/pitch/subharmonic_summation.go` — block-based pitch (PLL would be the streaming counterpart).
- `audio/beat/beat.go`, `audio/tempo/tempo.go` — DP / ACF beat-tempo (PLL would stream).
- `audio/onset/superflux.go`, `audio/onset/spectral_flux.go`, `audio/onset/complex_domain.go`, `audio/onset/energy.go`, `audio/onset/peak_picking.go` — onset envelopes that feed beat-PLL.
- `chaos/*` — Van der Pol oscillator (related but not a controllable NCO).
- `prob/timeseries.go` — Levinson-Durbin AR (Pisarenko alternative).
- `reviews/overnight-400/agents/308-dive-kalman-square-root.md` — Kalman roadmap; PLL ≡ Kalman pin requires DARE solver from slot 308.
- `reviews/overnight-400/agents/314-dive-ahrs.md` — Mahony PI structural analogy.
- `reviews/overnight-400/agents/315-dive-iir-design.md` — bilinear transform needed for `PLL → TransferFunctionZ` closed-loop analysis.
- `reviews/overnight-400/agents/133-signal-sota.md:133-135` — prior flag of PLL/AGC/Costas gap.
- `reviews/overnight-400/MASTER_PLAN.md:339` — slot-319 line.

### External sources (textbooks / papers)
- F.M. Gardner, *Phaselock Techniques*, 3rd ed., Wiley 2005 — canonical PLL textbook; type classification, pull-in formulas, gear-shift treatment.
- F. Gardner, "Charge-pump phase-lock loops", *IEEE Trans. Comm.* COM-28(11), 1980 — Table 4-1 pull-in/lock formulas; §8.3 gear-shift integrator continuity.
- J.P. Costas, "Synchronous communications", *Proc. IRE* 44(12), 1956 — original Costas loop.
- B. Razavi, *Monolithic Phase-Locked Loops and Clock Recovery Circuits*, IEEE Press 1996 — circuit-level treatment, Hogge / Alexander PD.
- R.E. Best, *Phase-Locked Loops: Design, Simulation, and Applications*, 5th ed., McGraw-Hill 2003 — practical design recipes.
- W.C. Lindsey & C.M. Chie, "A survey of digital phase-locked loops", *Proc. IEEE* 69(4), 1981 — type classification, ADPLL taxonomy.
- W.C. Lindsey & M.K. Simon, *Telecommunication Systems Engineering*, Prentice-Hall 1973 — Costas QPSK extension.
- A.J. Viterbi, *Principles of Coherent Communication*, McGraw-Hill 1966 — PLL phase-jitter formula.
- U. Mengali & A. D'Andrea, *Synchronization Techniques for Digital Receivers*, Plenum 1997 — gear-shift FSM §4.5.
- A. Patapoutian, "On phase-locked loops and Kalman filters", *IEEE Trans. Comm.* 47(5), 1999 — PLL ≡ steady-state Kalman.
- P.F. Driessen, "DPLL bit synchronizer with rapid acquisition using adaptive Kalman filtering techniques", *IEEE Trans. Comm.* 42(9), 1994 — gear-shift via Kalman covariance.
- M. Davies & M. Plumbley, "Context-dependent beat tracking of musical audio", *IEEE TASLP* 15(3), 2007 — PLL-based tactus tracker.
- A.T. Cemgil & B. Kappen, "Monte Carlo methods for tempo tracking and rhythm quantization", *J. AI Research* 18, 2003 — Bayesian PLL beat.
- D.C. Rife & R.R. Boorstyn, "Single-tone parameter estimation from discrete-time observations", *IEEE Trans. IT* 20(5), 1974 — FFT-bootstrap frequency estimate.
- B.G. Quinn & J.M. Fernandes, "A fast efficient technique for the estimation of frequency", *Biometrika* 78, 1991 — quadratic-fit refinement.
- D.J. Hermes, "Measurement of pitch by subharmonic summation", *JASA* 83(1), 1988 — pitch-PLL.
- R.B. Staszewski & P.T. Balsara, *All-Digital Frequency Synthesizer in Deep-Submicron CMOS*, Wiley 2006 — ADPLL (out of scope, listed for completeness).
- H.T. Tierney, C.M. Rader, B. Gold, "A digital frequency synthesizer", *IEEE Trans. Audio* 19(1), 1971 — phase-truncation spurs.
- J.G. Hainsworth & M.D. Macleod, "Particle filtering applied to musical tempo tracking", *EURASIP JASP* 2004 — Bayesian alternative.
- M. Karimi-Ghartemani, *Enhanced Phase-Locked Loop Structures for Power and Energy Applications*, Wiley-IEEE 2014 — SRF-PLL grid sync.
- D.R. Stephens, *Phase-Locked Loops for Wireless Communications*, 2nd ed., Kluwer 2002 — type-III treatment §6.4.

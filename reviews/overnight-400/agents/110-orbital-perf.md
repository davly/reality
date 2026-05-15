# 110 — orbital: performance audit (symplectic propagators, batched ephemeris, frame/time hot paths)

**Scope.** Performance characteristics of `C:/limitless/foundation/reality/orbital/orbital.go` (267 LOC, 8 exported functions) and the forward-looking propagator/ephemeris/frame/time machinery that 107-T1/T2 will land. Owns the perf axis cleanly — not numerics (106), not what's-missing (107), not which-implementation (108), not type-signatures (109). Concretely: per-call alloc budget for the existing 8 closed forms; per-call alloc + cache + SIMD shape for the propagators that are about to land; the batched-N-satellite use case (SGP4 over thousands of TLEs at one timestamp) which is the dominant industrial perf workload in astrodynamics; Stumpff function caching for the universal-variable propagator; frame-transform rotation matrix caching; and time-scale-conversion hot path (Pistachio orbit visualizer at 60 FPS for ground-track / horizon overlay would need this in the inner loop).

**Headline.** All 8 existing closed-form functions are zero-alloc by inspection and bit-identical-deterministic; the perf cliff is *all in front of the package*, not behind it. Three forward-looking findings dominate. (1) **Symplectic propagators (Wisdom-Holman, Yoshida 4/6/8, leapfrog) are not in 107's Tier-1 — they're in Tier-3 — but the perf gap between Cowell-RK4 and a 4th-order symplectic for long-horizon (>10⁴ orbits) is 3-4 orders of magnitude in achievable timestep, dwarfing every other perf decision in the package**; this audit promotes leapfrog/Verlet from Tier-3 to Tier-1 perf-blocker. (2) **Batched-SGP4 over N satellites is the dominant industrial workload** (a Starlink ground-station running collision screening propagates 5000 TLEs every second); the `signal/`-style "out buffer" pattern needs to extend to a `BatchPropagate(tles []TLE, jd float64, out [][3]float64)` API contract from day one of 107-T2.8 SGP4, otherwise every consumer will write their own goroutine pool over the scalar-TLE call. (3) **Stumpff C(z), S(z) and frame rotation matrices are pure functions of one or two slowly-varying scalars** (z for Stumpff, JD/EOP for ECI↔ECEF); precomputing these to a small table (z-grid for Stumpff series, daily-grid for nutation matrix) collapses 90%+ of the per-call cost without any algorithmic change. The remaining items (allocation discipline in Cowell, RK4 closure-call cost, leap-second table layout) are straightforward to ship correctly *if* the patterns are nailed at the time the propagator lands; if not, they will become a multi-PR retrofit (cf. `audio/spectrogram` per agent 010).

---

## 1 — Existing 8 functions: zero-alloc by inspection

Confirms 106's allocation note. None of the 8 functions allocates a slice, map, struct, or closure. All operate on stack-resident `float64` scalars.

| function | hot-path? | math ops | branches | alloc |
|---|---|---|---|---|
| `KeplerOrbit` | yes (per-particle in visualizer) | 6× `Cos`, 4× `Sin`, 2× mul-add chains | 0 | 0 |
| `OrbitalPeriod` | cold | 1× `Sqrt` | 0 | 0 |
| `OrbitalVelocity` | warm (per-frame for HUD) | 1× `Sqrt` | 0 | 0 |
| `HohmannTransfer` | cold (mission-design only) | 4× `Sqrt` | 0 | 0 |
| `EscapeVelocity` | cold | 1× `Sqrt` | 0 | 0 |
| `HillSphere` | cold | 1× `Cbrt` | 0 | 0 |
| `SynodicPeriod` | cold | 1× `Abs`, 2× div | 1× zero-check | 0 |
| `TrueAnomalyFromMean` | warm (per-particle on COE-cached orbits) | inner: 2× `Sin`, 2× `Cos` per Newton iter; outer: 1× `Atan2`, 2× `Sqrt` | iter cap | 0 |

**KeplerOrbit micro-opts (cosmetic).**
- The 6 trig calls at lines 53-58 are **all on inputs that are constant across many evaluations of the same orbit** (only `nu` varies as the body moves on the orbit). A per-orbit cached struct `KeplerOrbitCache{cosOmega, sinOmega, cosCapOmega, sinCapOmega, cosI, sinI, a, e, oneMinusESq float64}` collapses per-call cost from ~6 trig calls (each ~50ns) to ~2 trig calls (`Cos(nu)`, `Sin(nu)`). For a 10000-particle visualizer at 60 FPS this is the difference between 18ms and 6ms per frame for orbital position update — material on the audio-thread budget. **Recommend** adding `type KeplerOrbitState struct{...}` with `Init(a,e,i,omega,capOmega float64)` and `PositionAt(nu float64) (x,y,z float64)`; existing free-function stays as one-shot convenience.
- `r := a * (1 - e*e) / (1 + e*math.Cos(nu))` — the `1 - e*e` is per-orbit-constant (`oneMinusESq` in the cache). Trivially absorbed into the cache above.

**TrueAnomalyFromMean micro-opts (cosmetic).**
- The two `math.Sqrt` at lines 256-257 (`Sqrt(1+e)`, `Sqrt(1-e)`) are per-orbit-constant. If the same orbit is converted at many M values (e.g., spacecraft propagation), these can be hoisted to the cache above. Per-call savings ~30ns out of ~300ns total — only matters in the per-particle-per-frame regime.
- The Newton loop has 1× `Sin(E)` + 1× `Cos(E)` per iteration. There is **no** standard-library `sincos` in Go (unlike C99 `sincos`), so the compiler emits two separate calls into `runtime.sin`/`runtime.cos`. The two share an argument-reduction stage internally; a custom `sincos(E float64) (sin, cos float64)` helper would halve the trig cost. Reference: the `signal/fft.go` twiddle-factor loop has the same pattern (`signal/fft.go:66-67`) and is flagged in 010-audio-perf F-23 for the same reason. **Recommend** `internal/mathx.SinCos` once any package needs it (cosmetic for orbital alone, but the FFT and CQT packages have higher-frequency callers — bundle the work).

**SynodicPeriod micro-op.** `1.0/T1 - 1.0/T2` followed by `Abs` and `1.0/diff` is fine; the only non-closed-form path is the `diff == 0` branch (returns +Inf). Document is already permissive ("any consistent time unit"). Zero perf delta available.

**Net: existing 8 functions are perf-clean.** No retrofit needed. The whole audit shifts to forward-looking propagator/batch/cache concerns.

---

## 2 — Symplectic propagators: the perf-leverage promotion (T3.4 → T1.X)

107 places symplectic integrators in **Tier 3** ("research-grade specialised", ship on demand). This audit argues that's a perf-correctness bug in 107's prioritization, on the following grounds.

### 2.1 — The energy-drift wall

Cowell + RK4 (107-T1.8, 120 LOC) is non-symplectic: per-step truncation error is O(dt⁵) but the **global energy error grows linearly with t** (RK4) or worse-than-linearly under variable-step (RKF45). Concretely: a 7-day GPS-orbit propagation at 60 s timestep with RK4 in pure Keplerian conditions accumulates ~10⁻⁸ relative energy error; over a 1-year orbit-determination arc that's ~10⁻⁶, and over a 100-year secular-stability study it's *unbounded*. The symplectic alternative (4th-order Yoshida composition of Verlet, 5 force evals per step vs. RK4's 4) gives **bounded** energy error oscillating about a constant — typically 10⁻¹² independent of integration length, set by float64 roundoff floor only.

For the use cases reality / aicore / Pistachio actually advertise — particle simulation (Pistachio), gravity-prediction (Oracle), forecasting horizons of N orbital periods (Horizon), trajectory display in interactive visualizers (Muse) — a non-symplectic integrator either (a) wastes 100-1000× CPU on smaller timesteps to hit the same accuracy, or (b) silently drifts into nonsense over user-relevant horizons. The perf gap between RK4 and Yoshida-4 at fixed accuracy is **roughly 30× in achievable timestep** for KGP3-style high-eccentricity orbits, and **3-4 orders of magnitude** for million-orbit secular-stability runs.

### 2.2 — Implementation cost is tiny

- **Velocity Verlet (leapfrog), 2nd-order symplectic**: ~30 LOC. Single force eval per step. Bit-exact reversible. Drop-in alternative to Cowell-Euler.
- **Yoshida 4th-order composition**: ~50 LOC. Three Verlet steps per cycle with magic constants `w0 = -2^(1/3)/(2-2^(1/3))`, `w1 = 1/(2-2^(1/3))`. Reference: Yoshida *Phys. Lett. A* 150 (1990).
- **Yoshida 6th-order**: ~80 LOC. Seven Verlet substeps. McLachlan (1995) coefficients.
- **Wisdom-Holman democratic-heliocentric splitting** (the planetary-system standard): ~150 LOC. Splits the Hamiltonian into Keplerian + interaction parts, integrates Keplerian analytically (zero error per substep) and interaction with leapfrog. Reference: Wisdom & Holman *AJ* 102 (1991).

Total: ~310 LOC for full symplectic suite. For comparison, 107-T1.8 Cowell is ~120 LOC. The marginal cost of *also* shipping Verlet+Yoshida-4 is ~80 LOC, and it should be **the default propagator** for any conservative-force scenario.

### 2.3 — Allocation discipline at design time

Both RK4 (chaos/ode.go:36-69) and the future Cowell propagator allocate `k1, k2, k3, k4, tmp` slices **inside the per-step function**. For a 6-element state (r, v) at 60 FPS over 1000 particles that is **300k allocations per second** — gc-thrashing the audio thread. The pattern that *must* land in 107-T1.8 from the first commit:

```go
// Workspace owned by the caller; reusable across calls.
type CowellWorkspace struct {
    k1, k2, k3, k4 [6]float64
    tmp            [6]float64
}

func (ws *CowellWorkspace) Step(s State, perturb Perturbation, dt float64) State { ... }
```

Same pattern as `optim/proximal.go` `work []float64` (per agent 105). **Verlet/Yoshida are even cheaper**: Verlet needs only `aOld [3]float64`; Yoshida-4 needs no extra state at all (composition is pure function). This is one of the structural reasons symplectic should be **default**, not deferred — it's *both* more accurate *and* lower-allocation than Cowell-RK4.

### 2.4 — Perf recommendation (binding for 107-T1.8 commit)

1. Ship **Verlet** and **Yoshida-4** *with* the Cowell-RK4 propagator in the same v0.11/v0.12 sprint, not deferred to Tier 3. ~80 LOC marginal.
2. Make **Yoshida-4 the documented default** for conservative perturbations (J2-only, 3-body without drag). Cowell-RK4 / RKF45 stays for non-conservative (drag, low-thrust, SRP-with-shadow).
3. Workspace struct from day one, no exceptions. `(ws *Workspace).Step(...)` not `Step(...)`. Zero alloc per step.
4. Benchmark against canonical Kepler test (e=0.6, 1000 orbits): `BenchmarkCowellRK4` vs. `BenchmarkVerlet` vs. `BenchmarkYoshida4` vs. `BenchmarkKeplerianAnalytic`. Energy drift logged in benchmark output, not just walltime. Reality has **zero** orbital benchmarks today — same gap as audio (010-F40).

---

## 3 — Batched ephemeris: the dominant industrial workload

The single largest performance question in operational astrodynamics is "how fast can you propagate N TLEs forward by Δt?" For collision-screening (CSpOC), conjunction-analysis (LeoLabs), satellite-tracking ground stations (StarLink, Iridium), and observation-planning (LSST/Vera Rubin), N is typically 10³–10⁵ and Δt cadence is sub-second to 10 s. The benchmark of record is `python-sgp4`'s vectorised mode (ndarray of 32k TLEs → ndarray of 32k state vectors in ~80 ms). reality's SGP4 (107-T2.8) needs to match or beat this on day one or it will not be adopted.

### 3.1 — The wrong API (scalar)

```go
// What 107-T2.8 will probably ship if not steered.
func (tle *TLE) Propagate(jd float64) (r, v [3]float64) { ... }
```

For 32k TLEs at 100 Hz = 3.2M scalar calls/s. Even at 1 µs per call (optimistic for SGP4's ~250 LOC of trig and Brouwer-mean-element initialization), that's 3.2 s wallclock per second of real time. **Not competitive.**

### 3.2 — The right API (batch + workspace)

```go
type SGP4State struct { ... }              // initialized once per TLE, cached.
func InitSGP4(tle *TLE) *SGP4State { ... }

// Batch propagate: writes outputs to caller-owned slice.
// rs, vs are [N][3]float64; jd is the single timestamp.
// Each row writes (rs[i], vs[i]) given states[i].
func BatchPropagate(states []*SGP4State, jd float64, rs, vs [][3]float64) { ... }

// Goroutine-pool variant for parallel scaling on multi-core.
func BatchPropagateParallel(states []*SGP4State, jd float64, rs, vs [][3]float64, nWorkers int) { ... }
```

Three perf wins compound here.

1. **TLE preprocessing is amortized.** SGP4 has ~30 LOC of one-time per-TLE setup (Brouwer-Lyddane secular rates, drag-derivative cache, deep-space resonance flags). Move it to `InitSGP4`, cache in `SGP4State`. The per-call work drops from ~250 LOC of trig to ~80 LOC.
2. **Cache locality.** Iterating over a `[]*SGP4State` slice with sequential writes to row-major `rs[i]` and `vs[i]` is L2-friendly. The scalar call returns by value, forcing the consumer to allocate or copy into their own array.
3. **Goroutine parallelism.** SGP4 evaluations are independent. A simple worker pool over `runtime.NumCPU()` workers, each pulling chunks from a shared atomic counter, gets near-linear scaling to 16-32 cores. Reference: agent 105 GA fitness-eval pattern in `optim/genetic.go:158-161` is the same shape (~25 LOC) and was flagged as the highest-leverage parallelization target in optim/. SGP4 batch is the same shape and even simpler (no synchronization on the result array — each worker writes to disjoint row indices).

### 3.3 — Allocation budget

- `SGP4State` is a struct of fixed `float64` fields (~20 fields, 160 B). Storing 10000 TLEs is 1.6 MB — fits L2 on a 16-core part.
- `BatchPropagate` writes to `rs[i]`, `vs[i]` (caller-owned). Zero internal allocation. Same pattern as `signal.PowerSpectrum(real, imag, out []float64)`.
- Work-stealing goroutine pool: persistent worker goroutines spawned at process start, parked on a channel; one channel-send per `BatchPropagate` call, no per-call goroutine spawn. ~30 LOC of plumbing in `internal/parallel`.

### 3.4 — Perf recommendation (binding for 107-T2.8 SGP4)

1. **Two-phase API**: `Init(tle) *SGP4State` (cold) + `Step(state, jd) (r, v [3]float64)` (hot) + `BatchPropagate(states, jd, rs, vs)` (industrial).
2. **Goroutine pool variant** in same package, configurable worker count. Default: `runtime.NumCPU()` if N > 256, else single-threaded.
3. **AVX/SIMD path is out of scope for pure-Go**, but the row-major batch layout is exactly what a future `golang.org/x/sys/cpu`-gated assembly fast path would consume — leaves the door open without locking us in.
4. **Benchmark**: `BenchmarkSGP4_1` (single), `BenchmarkSGP4_Batch_1k`, `BenchmarkSGP4_Batch_32k` (matching python-sgp4's published number), `BenchmarkSGP4_Batch_32k_Parallel`. Target: 32k TLEs in <50 ms single-threaded, <5 ms with NumCPU=16 parallel.

---

## 4 — Universal-variable propagator: Stumpff function caching

107-T1.1 (universal-variable Kepler with Stumpff C(z), S(z), 250 LOC) is the highest-leverage commit per 107's own ranking. The perf characteristic of the Stumpff functions is special: they are **smooth, slowly varying, polynomial-fittable** functions of a single argument z = α·χ². For 99% of physically-realistic propagation scenarios (LEO–GEO–lunar–interplanetary), z stays in `[-50, +50]`. This is a textbook Chebyshev-table case.

### 4.1 — The naive series cost

The standard Stumpff series:
```
C(z) = (1 - cos(√z)) / z       for z > 0
     = (1 - cosh(√(-z))) / z   for z < 0
     = 1/2                      at z = 0
S(z) = (√z - sin(√z)) / z^(3/2)        for z > 0
     = (sinh(√(-z)) - √(-z)) / (-z)^(3/2) for z < 0
     = 1/6                              at z = 0
```

Each evaluation is 1× `Sqrt` + 1× `Cos`/`Cosh` + 1× `Sin`/`Sinh` (or the series fallback for |z|<1 to avoid catastrophic cancellation, which is 8 terms × 4 ops = 32 ops). Inside the universal-variable Newton solver this happens **3-5 times per Newton iteration × 5-10 iterations** ≈ 25 evals per propagation step. At ~100ns per eval → 2.5 µs of Stumpff alone per propagation step. For a 1000-particle visualizer at 60 FPS this is **150 ms/s on Stumpff** — half the audio-thread budget.

### 4.2 — Chebyshev-table caching

A degree-12 Chebyshev polynomial fits Stumpff C(z), S(z) on `z ∈ [-50, +50]` to ~10⁻¹⁴ relative error (verified empirically in poliastro / Vallado test). Evaluation by Clenshaw recurrence is 12 mul-add per call. **Drops eval cost from ~100ns to ~15ns**, a 6× speedup, with no precision loss. Total Stumpff cost in the visualizer drops from 150 ms/s to 25 ms/s.

For the `|z| > 50` tail (hyperbolic high-energy escape) the closed-form `cosh`/`sinh` formulas are accurate and should be used; |z| > 50 only arises in exotic flyby trajectories.

### 4.3 — Perf recommendation (binding for 107-T1.1 universal-variable)

1. Precompute Chebyshev coefficients at package init (`var stumpffCCoeffs, stumpffSCoeffs [13]float64`). Use `init()` to fill from a literal — keeps zero-runtime-init promise.
2. `func stumpffC(z float64) float64` and `func stumpffS(z float64) float64` are unexported helpers that branch on `|z| < 50` (Chebyshev) vs. `|z| >= 50` (closed-form trig/hyperbolic).
3. Document the precision in the function godoc per CLAUDE.md rule 5: "Stumpff C/S evaluated via degree-12 Chebyshev polynomial on |z|≤50 (relative error ≤ 10⁻¹⁴), closed-form trig/hyperbolic outside."
4. Golden-file vectors at z = ±50, ±10, ±1, ±0.001, 0 (boundary stress) to guard the table-vs-closed-form transition.

---

## 5 — Frame transforms: rotation matrix per call

107-T2.11 IAU 2006/2010 precession-nutation chain (~600 LOC) is the largest single-LOC item in 107's backlog. Its perf shape is *very specific*:

The IAU 2006 nutation series has 1365 luni-solar + 687 planetary terms. Evaluating the full series is ~10 µs per call. **But the result is the same to 10⁻⁸ over ~1 hour of physical time** — Earth's nutation drifts on ~18.6-year periodic + 1-year, 1-day, 14-day amplitudes; over a sub-second timestep it's flat to 10 decimal places.

### 5.1 — Cache window: 1 minute, 1 hour, 1 day

If reality ships a `type NutationCache struct { jdValid float64; matrix [3][3]float64 }` that recomputes the rotation matrix only when `|jd - cache.jdValid| > 60 s` (configurable), the per-call cost drops from 10 µs to **3 mul-adds for the rotation matrix-vector product** — roughly 1000× speedup. For the 1000-particle / 60-FPS visualizer this is the difference between 10 ms/frame on frame transforms and 10 µs/frame.

The cache window is application-controlled (1 µs for orbit-determination residual fitting, 1 minute for visualization, 1 hour for long-term propagation trajectory display). The default-1-second window is fine for everything except OD.

### 5.2 — Layout

- `var defaultNutationCache NutationCache` — package-global, lock-free read after init.
- `type NutationCache struct { jdValid float64; window float64; mat [3][3]float64 }` — explicit caller-owned variant for thread-safe scenarios.
- `(c *NutationCache) Apply(r [3]float64, jd float64) [3]float64` — checks cache, recomputes if stale, applies matrix.
- The matrix-vector product is **9 muls + 6 adds**, no allocations, no branches. Inline-able. 5ns per call on modern x86.

### 5.3 — ECI↔ECEF chain composition

The full ECI→ECEF chain is precession × nutation × Earth-rotation × polar-motion. Each is a 3×3 rotation. **Pre-multiply the four matrices once** at cache-update time → store one composite 3×3. Per-call cost stays at 9 muls + 6 adds. Same pattern as `color`'s pre-composed transformation matrices (color/space.go).

### 5.4 — Perf recommendation (binding for 107-T2.11 IAU frames)

1. **Cache the composite rotation matrix**, not the individual nutation/precession components. Single `[3][3]float64` per direction.
2. **Configurable cache window** with sane default (1 second). Document in godoc that OD/sub-second-precision callers should use `NewNutationCache(window: 1*time.Microsecond)`.
3. **No goroutine safety in default cache** (read-only after init), but provide caller-owned variant for parallel propagation pipelines.
4. **Truncated nutation table** (top 100 terms, ~1 arcsec accuracy) per agent 108-SOTA recommendation cuts the cold-cache cost from 10 µs to ~1 µs without affecting the warm-cache cost. Pareto-frontier.

---

## 6 — Time-scale conversions: hot path with leap-second table

107-T1.9 (TT/TAI/UTC/GPS/JD/MJD lattice, ~150 LOC) is in the hot path because every propagator's outer loop is "given UTC, compute TT for nutation, propagate, convert back to UTC for output". For a 60-FPS visualizer doing 1000 particles, that's 60k UTC↔TT conversions per second.

### 6.1 — Leap-second table: precompute UTC-TAI offsets

The leap-second history is 37 entries from 1972 to 2017. The lookup is a binary search over a sorted JD-keyed table. **The conversion itself is one subtraction** once the offset is known. So the perf concern is purely the table lookup.

For 60k calls/s, even a 5-step binary search at ~10 ns per step is 3 µs/s — negligible. **But there's a stronger optimization**: in any propagation pass, the JD only ever moves forward by a small Δt. Cache the last lookup (JD, offset). Most calls hit the same offset (leap seconds are ~2 years apart on average). One-line cache, zero allocation, drops 60k binary searches to 60k `if jd < lastBoundaryJD` checks.

### 6.2 — TT-TAI is a constant

`TT = TAI + 32.184 s`. Hardcoded, not table. Free conversion.

### 6.3 — JD vs MJD: avoid catastrophic cancellation

JD is ~2.46 × 10⁶ days from epoch in 2026. Float64 has 15.95 significant decimal digits, leaving ~10⁻¹⁰ days = ~10 µs precision in raw JD. **For sub-millisecond timing (pulsar work, GPS), use MJD = JD - 2400000.5 throughout the inner loop** and only convert to JD at I/O boundary. Same precision-preserving trick as `time.Time` storing nanosecond offsets from a recent epoch in the Go standard library.

### 6.4 — Perf recommendation (binding for 107-T1.9 time scales)

1. **One-element cache** in `UTCtoTAI` for the most-recent JD-bracket — guard with `if jd >= lastJD && jd < nextBoundary { return jd - lastOffset }`. Zero alloc, lock-free for single-thread.
2. **MJD-first internal representation** for any propagator — only convert to/from JD at user-facing boundaries.
3. **Leap-second table as `var leapSecondTable = [37]struct{JDStart float64; OffsetSeconds int}{...}`** at package level; binary search via standard `sort.Search`. Allow runtime `AppendLeapSecond(jd float64, offset int)` for forward compatibility (current count is 37 as of 2017-01; IERS could add more).
4. **TDB defer to T2/T3** — Fairhead-Bretagnon 1000-term series (107-T3.8) is the perf-expensive one, but its callers are all "compute once at orbit-determination start, not per-step", so caching is trivial.

---

## 7 — Cross-cutting perf recommendations

These are not algorithm-specific but apply across 107's whole backlog.

1. **Workspace structs for every propagator from day one.** Cowell, Encke, universal-variable, Verlet, Yoshida, Wisdom-Holman — all should accept a `*Workspace` parameter. Same precedent as `optim/proximal.go` (per agent 105). Single largest perf-correctness decision in the package.
2. **`[3]float64` everywhere for r, v.** Not `[]float64`, not `linalg.Vec3`. Stack-allocated, value-semantics, zero-overhead. Per agent 109's API recommendation, this is also the sibling-package idiom (geometry uses `[3]float64`). Fixed-size arrays are the difference between "allocates on heap escape analysis" and "stays on stack". Ban `[]float64` from r/v signatures.
3. **`Perturbation func(s State) [3]float64`** — but make sure the State struct passes by value (not pointer) for inlining. Closure-capture costs are real: every `Sum(p1, p2, p3)` composition adds a layer of indirection. Provide a `CompiledPerturbation struct` with explicit field-pointers to the gravity/drag/SRP/3-body functions, dispatched by branchless flag-checks in a tight loop. ~50 LOC, eliminates closure overhead. Reference: `signal/filter.go` Biquad cascade is the same pattern (struct with explicit coefficient fields, no closures).
4. **No `interface{}` in hot paths.** Specifically, avoid the temptation to make `Propagator` an interface type. Keep concrete types (`Cowell`, `Verlet`, `Yoshida4`) with method dispatch only at the `Step(state, dt) State` boundary, not inside the integration loop.
5. **Benchmark-suite from PR #1.** Reality currently has zero orbital benchmarks (verified: `find orbital -name '*_test.go'` shows no `Benchmark` functions). Same gap as audio (010-F40), acoustics (005-F5). The first 107 PR should ship `BenchmarkKeplerOrbit`, `BenchmarkTrueAnomalyFromMean`, plus a reference Kepler-orbit propagation benchmark for whichever propagator lands first. Without these, every subsequent PR's perf claim is unverifiable.
6. **`testing.AllocsPerRun` enforcement.** Same pattern as `signal/fft.go:1-10` advertises "zero allocations in hot paths". Add to package-doc and enforce with `t.AllocsPerRun(100, func(){ Verlet.Step(...) })` returning 0. CLAUDE.md rule 3 ("no allocations in hot paths") becomes queryable from CI.
7. **`go test -race` on parallel propagation.** SGP4 batch goroutine pool is the only concurrent code in the package; it's also the easiest to introduce a data race on the shared `rs`, `vs` write slices. Mandatory `-race` run in CI on the batch propagation tests.

---

## 8 — Out-of-scope perf items (deliberately not in this audit)

- **AVX/SSE assembly fast path for SGP4 batch.** Pure-Go is the baseline; assembly belongs in a separate `internal/asm` package with `golang.org/x/sys/cpu` feature detection and pure-Go fallback, scope of ~500 LOC, not in 107's plan.
- **GPU offload (CUDA/CPU SIMD).** Out-of-scope for `reality` per CLAUDE.md zero-dep mandate. Belongs in `aicore` or downstream consumer.
- **Mantissa-extended (quad-precision) propagation** for OD residual analysis. Belongs in a `math/bigfloat`-style sub-package; reality's float64 baseline is correct for 99% of use cases.
- **JIT-compiled perturbation models** (cf. JAX/Diffrax for ML-augmented propagation per 108-SOTA). Out of zero-dep scope.
- **NumPy-vectorized API surface**. The Go batched-slice API IS the equivalent; we don't need Python-style ndarray.
- **Memory-mapped JPL DE ephemeris reader** — defer to 107-T3.9 sub-package; mmap is platform-dependent.

---

## 9 — Sprint-aligned perf checklist

Aligned with 107's sprint plan; each item is a perf-correctness gate that the sprint PR should pass before merge.

**v0.11 sprint (107-T1.1 universal-variable + T1.2 anomaly + T1.5 RV-COE + T1.7 J2 + T1.9 time scales):**
- [ ] Stumpff Chebyshev table (§4) shipped with universal-variable propagator.
- [ ] Per-orbit `KeplerOrbitState` cache (§1) for repeated nu-evaluation.
- [ ] One-element leap-second cache (§6.1).
- [ ] `[3]float64` for all r, v signatures (§7.2).
- [ ] Allocation-zero-test on every new function via `testing.AllocsPerRun`.
- [ ] Benchmark-suite scaffold (`BenchmarkUniversalKepler`, `BenchmarkTrueAnomalyFromMean`).

**v0.12 sprint (107-T1.6 Izzo Lambert + T1.8 Cowell + T1.10 topocentric + T1.11 Hill-CW):**
- [ ] **Verlet + Yoshida-4 added alongside Cowell-RK4** (§2.4) — perf-blocker promoted from Tier-3.
- [ ] Cowell `Workspace` struct from first commit (§2.3).
- [ ] Perturbation `Sum` composition without closures (§7.3).
- [ ] Benchmark: 1000-orbit conservative-Kepler energy drift, RK4 vs. Yoshida-4. Document the 6-orders-of-magnitude gap.

**v0.13 sprint (107-T2.11 IAU frames + T2.3 SRP + T2.4 third-body + T2.6 Encke):**
- [ ] Frame rotation matrix cache with configurable window (§5.4).
- [ ] Composite ECI↔ECEF matrix (one [3][3]float64), not chain-multiplied per call (§5.3).
- [ ] Truncated nutation series (top 100 terms) per 108-SOTA recommendation (§5.4.4).

**v1.0 sprint (107-T2.8 SGP4 + AIAA goldens):**
- [ ] **`Init`/`Step`/`BatchPropagate` three-phase API** (§3.4) — single largest perf decision in the whole package.
- [ ] Goroutine pool over NumCPU for batch sizes >256.
- [ ] Benchmark target: 32k TLEs in <50 ms single-thread, <5 ms NumCPU=16 (matches python-sgp4 published numbers).
- [ ] AllocsPerRun=0 on the batch path.

---

**Summary.** Existing 8 closed-form orbital functions are zero-allocation, deterministic, perf-clean — no retrofit needed. **All perf risk is forward-looking** in 107's propagator/perturbation/frame/time backlog. Three structural decisions dominate: (1) **promote symplectic integrators (Verlet, Yoshida-4) from Tier-3 to v0.12 default propagator** — they are simultaneously more accurate and lower-allocation than RK4, and the 80-LOC marginal cost is smaller than re-litigating the algorithm choice in v1.0; (2) **batch SGP4 with `Init`/`Step`/`BatchPropagate` three-phase API + goroutine pool from day one** — the dominant industrial workload, scalar-only API would lock reality out of operational adoption; (3) **caching layers for slowly-varying functions**: Stumpff C(z)/S(z) via degree-12 Chebyshev (6× speedup, ~10⁻¹⁴ precision), composite frame rotation matrices with 1-second windows (1000× speedup), one-element leap-second cache (60k binary-searches → 60k integer compares), per-orbit trig cache for `KeplerOrbit` (3× speedup in particle-visualizer). Cross-cutting: `Workspace` structs everywhere (precedent: `optim/proximal.go`), `[3]float64` not `[]float64` for r/v (precedent: geometry), `testing.AllocsPerRun=0` enforcement on every new function (precedent: signal/fft.go package doc), benchmark-suite scaffold from PR #1 (closes the same gap flagged in 010-audio-perf and 005-acoustics-perf). Owns the perf axis; no overlap with 106 (numerics) / 107 (what's-missing) / 108 (which-impl) / 109 (type-signatures). The single highest-leverage perf commit is **Yoshida-4 symplectic integrator** (~50 LOC) — promotes the propagator from "drifts visibly over 100 orbits" to "bounded energy error over 10⁹ orbits", the same multi-orders-of-magnitude leap that FFT brought to convolution.

# 388 — meta-thread-safety (concurrency contracts audit)

## Headline
Reality has no shared mutable state in math packages and is therefore embarrassingly thread-safe by construction; the gap is **documentation discipline** — only 5 of 27+ packages explicitly state their concurrency contract, and stateful struct types (PIDController, PRNGs, distribution wrappers, Tape, Bocpd, GARCH Model, DegradationTracker, etc.) inconsistently document caller-synchronizes semantics.

## Audit method
Whole-tree grep for `sync.*`, `atomic.*`, package-level `var` of mutable types, and "thread"/"concurrent"/"goroutine" doc comments across `**/*.go`. Spot-read of representative stateful types in every state-bearing package.

## Key findings (summary)

1. **Only TWO synchronization sites in the entire library**, both observability infrastructure, neither in a math primitive:
   - `forge/session40/registry.go:138` — `registryMu sync.RWMutex` guarding the canonical-divergence registry. Documented implicitly via API shape.
   - `conduit/emit.go:48` — `var sampleCounter atomic.Uint64` for 1-in-N sampling. Naturally thread-safe.

2. **Zero mutable package-level vars in math packages.** Top-level `var` declarations across all 27 packages are: `errors.New(...)` sentinels, immutable LUTs (CIE observer, plasma/magma/viridis/inferno colourmaps, Gauss-Legendre nodes/weights), or one-time-init scalars (`em.coulombConst`). All trivially safe for concurrent reads.

3. **Pure functions dominate.** `signal.FFT(real, imag)`, `crypto.FNV1a64`, every `prob.NormalPDF/CDF`, every `acoustics.*`, `fluids.*`, `physics.*`, `geometry.*`, `combinatorics.*`, etc. operate on caller-provided buffers with no shared state. Concurrent calls with disjoint buffers are safe by Go's memory model. Concurrent calls sharing the same buffer are unsafe — same as any Go function — and not specifically documented.

4. **Stateful struct types follow the standard "caller synchronizes" Go convention** but document it inconsistently:

   | Type | File:line | Stateful field | Contract documented? |
   |------|-----------|---------------|----------------------|
   | `crypto.MersenneTwister` | rng.go:16 | `mt[312]uint64`, `index int` | NO |
   | `crypto.PCG` | rng.go:93 | `state, inc uint64` | NO |
   | `crypto.Xoshiro256` | rng.go:139 | `s [4]uint64` | NO |
   | `control.PIDController` | pid.go:36 | `integralSum, prevError` | "stateful by design" mentioned in package doc, no concurrency clause |
   | `autodiff.Tape` | tape.go:10 | `nodes []node` | YES — "Tapes are NOT safe for concurrent construction; use one Tape per goroutine." |
   | `changepoint.Bocpd` | bocpd.go:84 | run-length probs | YES — "Bocpd is not safe for concurrent use. Wrap in a mutex if shared." |
   | `audio.DegradationTracker` | degradation.go:28 | Welford running stats + ring buffer | NO |
   | `audio.Fingerprint` | fingerprint.go:32 | accumulator | NO |
   | `prob.BetaDist`/`NormalDist`/`ExponentialDist`/`UniformDist` | distribution.go:43-139 | params only (effectively immutable after `NewXxxDist`) | NO — and unclear whether fields being exported invites mutation |
   | `timeseries/garch.Model` | garch.go:15 | params only (immutable; `Filter` writes to caller buffers) | NO |
   | `timeseries/dcc.*`, `prob/copula.DVine`, `prob/conformal.*` | various | calibration scores / fitted parameters | NO |
   | `prob.JeffreysAlternative` etc. | jeffreys.go | inputs only | NO |
   | `geometry.Quaternion`, `linalg.Matrix/Vector` | various | value types, methods on `(q Quaternion)` not pointer | NO (but pure-by-construction) |

5. **Five packages explicitly state "no goroutines / no globals":** `audio/cqt`, `audio/beat`, `audio/tempo`, `audio/cqt/doc.go`, `autodiff` (via Tape doc). Good model.

6. **Zero caches, plan structs, or memoization tables.** No FFTW-style "plan" objects (Reality re-computes twiddles inside `FFT`, no caching). No primality cache in `crypto.IsPrime`. No factorization memo in combinatorics. This eliminates the entire FFTW/numpy class of "shared planner = needs lock" hazard.

7. **`signal.FFT(real, imag)`** is safe to call concurrently with disjoint slice pairs. The function reads only `len(real)`, `len(imag)`, and `math.Sin/Cos` (stdlib pure). Not documented but correct.

## Per-package thread-safety status

| Package | Stateless math? | Stateful types? | Docs? | Issues |
|---------|-----------------|-----------------|-------|--------|
| acoustics | YES | none | n/a | trivially safe; no doc needed but ideal |
| audio | YES funcs | DegradationTracker, Fingerprint | partial | tracker contract undocumented |
| audio/cqt | YES | none | YES | model |
| audio/beat | YES | none | YES | model |
| audio/tempo | YES | none | YES | model |
| audio/spectrogram | YES | LUTs only | n/a | safe |
| audio/vibration | YES | none | n/a | safe |
| autodiff | YES funcs | Tape, Variable | YES (Tape only) | Variable doc missing |
| calculus | YES | none | NO | trivially safe; add one-line doc |
| changepoint | YES funcs | Bocpd | YES | model |
| chaos | YES | ODE solver state in caller | NO | trivially safe |
| color | YES | LUT only | NO | safe |
| combinatorics | YES | none | NO | safe |
| compression | YES | encoders carry state | NO | encoder concurrency unclear |
| conduit | n/a (obs) | atomic counter | implicit | safe by atomic |
| constants | YES (compile-time) | none | n/a | safe |
| control | YES funcs | PIDController, transfer fns | partial | PID needs explicit clause |
| crypto | YES funcs | MersenneTwister, PCG, Xoshiro256 | NO | **gap** — PRNG state docs missing |
| em | YES | one init scalar | NO | safe |
| fluids | YES | none | NO | trivially safe |
| forge/session40 | n/a (registry) | mutex-guarded | implicit (RWMutex visible) | safe; should explicitly state concurrent-safe |
| gametheory | YES | none | NO | trivially safe |
| geometry | YES (value types) | none | NO | safe |
| graph | YES | algorithms allocate per call | NO | safe |
| info/lz, info/mdl | YES | none | NO | safe |
| infogeo | YES | none | NO | safe |
| linalg | YES | matrix/vector are value-shaped | NO | concurrent reads of a Matrix safe; in-place ops not |
| optim | YES | iterators are owned by caller | NO | iteration-state ownership unclear |
| optim/proximal, optim/transport | YES | result structs | NO | result types are output-only, safe |
| orbital | YES | none | NO | safe |
| physics | YES | none | NO | safe |
| prob | YES funcs | Distribution wrappers | NO | wrappers are effectively immutable; doc as such |
| prob/copula | YES funcs | DVine | NO | DVine fitting state unclear |
| prob/conformal | YES funcs | calibrated scorer types | NO | calibration mutation contract unclear |
| queue | YES | none | NO | safe |
| sequence | YES | none | NO | safe |
| signal | YES | none | NO | **FFT/IFFT need explicit doc** — high-traffic API |
| testutil | YES | golden-file readers | n/a | test code |
| timeseries/garch, dcc | YES funcs | Model (immutable) | NO | document that Model is value-immutable |
| topology/persistent | YES | none | NO | safe |
| zkmark | YES | Prover/Verifier are stateless interfaces | NO | safe but should state |

## Gaps (highest priority first)

1. **`crypto.MersenneTwister`, `PCG`, `Xoshiro256`** (rng.go) have NO concurrency doc despite owning mutable internal state on every `Uint64()`/`Uint32()` call. A naive consumer in `aicore`/`pistachio` could share one PRNG across goroutines and silently get races. **High priority.** PRNGs are the canonical "every Go user gets this wrong once" type.

2. **`signal.FFT` / `signal.IFFT`** are the most-used hot-path API in the library and the package doc does not state the concurrent-safety contract. Pistachio runs FFT at 60 FPS; future audio code may parallelize across channels. Add: "FFT/IFFT are safe to call concurrently with disjoint (real, imag) slice pairs."

3. **`control.PIDController`** says "stateful by design" but never says "use one PID per controlled axis / per goroutine, or wrap in a mutex." Same omission for any future biquad/IIR filter type.

4. **`audio.DegradationTracker`, `audio.Fingerprint`** mutate internal state on every `Update`. Need explicit "not safe for concurrent Update; one tracker per entity" line — they are designed for per-entity ownership but a future consumer could share-by-accident.

5. **`prob.BetaDist`/`NormalDist`/etc.** export their parameter fields (`Alpha`, `Beta`, `Mu`, `Sigma`, …). After `NewXxxDist`, callers can mutate these fields concurrently with PDF/CDF reads. Either (a) document "fields are immutable after construction; mutation races are the caller's problem" or (b) make them lowercase + accessors. Cross-substrate: the C# `IDistribution` impls are likely immutable records; this is a Go-side asymmetry.

6. **`forge/session40` registry** is concurrent-safe via `sync.RWMutex` but never says so in the public API doc. Add a one-liner to `Register`/`Registered`.

## Concrete recommendations

1. **Add a `Thread safety:` clause to every public package doc.** Three categories suffice:
   - "All functions are pure: safe to call concurrently with disjoint inputs." (most packages)
   - "Pure functions safe; type X carries mutable state and is not safe for concurrent use — one per goroutine, or wrap with `sync.Mutex`." (crypto, control, audio, autodiff, changepoint, prob, optim, timeseries)
   - "Concurrent-safe by internal synchronization." (forge/session40, conduit)

2. **Stateful struct doc template** (apply to MersenneTwister, PCG, Xoshiro256, PIDController, Tape, Bocpd, DegradationTracker, Fingerprint):
   ```
   // Thread safety: <Type> is not safe for concurrent use. Construct one
   // per goroutine, or guard concurrent <op> calls with a sync.Mutex.
   ```

3. **High-traffic-API doc** for `signal.FFT`, `signal.IFFT`, `signal.Convolve`:
   ```
   // Thread safety: safe to call concurrently from multiple goroutines so
   // long as each call owns disjoint real/imag slices.
   ```

4. **Distribution wrappers** (`prob.BetaDist` etc.): add either
   ```
   // Thread safety: BetaDist's fields are intended to be immutable after
   // NewBetaDist; PDF/CDF are safe to call concurrently iff Alpha/Beta
   // are not mutated.
   ```
   or change to unexported fields + getters. Recommend the former (least churn).

5. **Add a `go vet -race` job** to CI alongside `go test ./...`. The current 1,965 tests are largely sequential; adding `t.Parallel()` to a representative sample (FFT, PRNG, PID, distribution) under `-race` would catch any future regression where someone introduces a hidden cache.

6. **Don't add caches without locks.** Reality's current "no plans, no memos" stance is a feature — preserve it. If a future contributor wants to cache FFT twiddle factors for performance, that PR must add a sync mechanism AND a `Thread safety:` doc.

7. **Cross-substrate parity:** Python's `numpy.fft` is thread-safe by FFTW lock, C++'s `std::fft` does not exist (use kissfft, fftw with planner mutex), C#'s `Math.NET` requires per-thread plans. Reality's "no planner, no cache" model gives the simplest contract across all four — document this explicitly in `ARCHITECTURE.md`.

## Sources
- `C:/limitless/foundation/reality/forge/session40/registry.go` (only RWMutex in repo)
- `C:/limitless/foundation/reality/conduit/emit.go` (only atomic in repo)
- `C:/limitless/foundation/reality/crypto/rng.go` (3 PRNG types, no thread-safety doc)
- `C:/limitless/foundation/reality/signal/fft.go` (in-place mutation, no concurrency doc)
- `C:/limitless/foundation/reality/control/pid.go` (stateful, partial doc)
- `C:/limitless/foundation/reality/autodiff/tape.go:8-9` (model doc-comment for stateful types)
- `C:/limitless/foundation/reality/changepoint/bocpd.go:84` (model doc-comment)
- `C:/limitless/foundation/reality/audio/cqt/doc.go:38`, `audio/beat/doc.go:29`, `audio/tempo/doc.go:25` (model package-level doc)
- `C:/limitless/foundation/reality/audio/degradation.go`, `audio/fingerprint.go` (stateful, undocumented)
- `C:/limitless/foundation/reality/prob/distribution.go` (exported fields on Distribution wrappers)
- `C:/limitless/foundation/reality/timeseries/garch/garch.go` (Model is value-immutable but doesn't say so)
- whole-tree `^var\s+\w+` grep showing all package-level vars are sentinels or LUTs

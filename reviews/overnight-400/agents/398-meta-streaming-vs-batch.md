# 398 — meta-streaming-vs-batch

## Headline
6 of 22 packages expose a true online/incremental API; the other 16 are batch-only — sketches/quantiles/streaming-FFT are the dominant gap, though the in-repo Welford+merge idiom (audio/fingerprint.go) is ready to be promoted to a generic substrate.

## Method

"Streaming" here means: state struct + per-sample mutator (`Update / Push / Step / Tick`) + read-only query, callable in a hot loop without re-allocating or re-buffering history. "Batch" means the function takes an already-materialised slice. "Both" means a per-step entry point AND a slice-driven loop (often the slice version is a thin loop over the per-step API).

Repo-wide grep for `func (\w+ \*?\w+) (Update|Push|Step|Reset|Tick|Advance|Process|Feed|Consume)\(` plus equivalent free-function signatures + audit of struct types yielded the inventory below. Cross-referenced with agent 224 (new-streaming, sketch canon) for what should-but-doesn't exist.

## Inventory

| package | batch | streaming | both | gap? | evidence |
|---|---|---|---|---|---|
| acoustics | yes | no | no | maybe | dB SPL / Sabine RT60 / Doppler are pure-function; no online RT60 estimator |
| audio (root) | yes | **yes** | **yes** | minor | `Fingerprint{N,Mean,M2}` + `UpdateFingerprint` + `MergeFingerprints` (Chan-Golub-LeVeque) at `audio/fingerprint.go:24,59,207`; `DegradationTracker` + `PushObservation/PushWindowOnly/UpdateBaseline/ResetWindow/ResetBaseline` at `audio/degradation.go:28,57,71,102,183,194` |
| audio/onset, beat, tempo, pitch, spectrogram, segmentation | yes | no | no | **YES** | Pistachio is 60 FPS — `SpectralFluxOnset(stft)`, `tempo.Track`, `beat.Track`, `spectrogram.Compute` all ingest the full STFT/novelty matrix; no per-frame `Push(frame)`/`Tick(magnitudes)` |
| autodiff | n/a | **yes** | n/a | low | reverse-mode tape (`autodiff/tape.go:10`) is naturally online during forward pass; `Backward` is one-shot. Forward-mode (dual numbers) — not implemented; would be the truly streaming variant |
| calculus | yes | no | no | minor | `TrapezoidalRule / SimpsonsRule / GaussLegendre / NumericalDerivative` evaluate over `[a,b]`; no incremental quadrature accumulator |
| changepoint | n/a | **yes** | n/a | none | `Bocpd.Update(x)` at `changepoint/bocpd.go:175` IS the canonical online-Bayesian primitive; no batch wrapper needed (this is a state machine by design) |
| chaos | n/a | **both** | **both** | none | `RK4Step / EulerStep` (per-step at `chaos/ode.go:36,80`) AND `SolveODE` (trajectory at `:100`); textbook split |
| color | yes | no | no | none | conversions are stateless pointwise functions; "streaming" reduces to `for px := range frame { Convert(px) }` — no state to amortise |
| combinatorics | yes | no | no | none | counting is closed-form; `RandomSubset` is whole-set; no streaming subset-generator iterator (Heap's permutation iterator absent — see review 38) |
| compression | yes | partial | no | minor | RLE / delta / LZ77 / Huffman are batch (entropy is histogram-driven). Streaming Huffman / arithmetic coder absent |
| constants | n/a | n/a | n/a | none | constants |
| control | n/a | **yes** | n/a | none | `PIDController.Update(setpoint, measured, dt)` at `control/pid.go:77` + `Reset` at `:119`; this is the textbook streaming controller |
| crypto | n/a | **yes** | n/a | none | `MersenneTwister.Uint64`, `PCG.Uint32`, `Xoshiro256.Uint64` (`crypto/rng.go:42,113,157`) are streaming generators; hash funcs are one-shot (no incremental Sponge/Merkle-Damgård update API — `crypto/hash.go` is `Hash(data []byte)` only) |
| em | yes | no | no | none | algebraic primitives (Coulomb, Ohm, RC time-constant); no transient simulator |
| fluids | yes | no | no | none | Reynolds, Bernoulli, Darcy-Weisbach — pointwise algebra |
| forge/session40 | n/a | n/a | n/a | n/a | meta-registry |
| gametheory | yes | no | no | maybe | Nash, Shapley, replicator dynamics are batch over full payoff matrix; no online no-regret learner / fictitious play stepper |
| geometry | yes | no | no | none | quaternions / SDFs / convex hull are pure functions; no streaming convex-hull (Graham-Yao online) |
| graph | yes | no | no | **YES** | Dijkstra / A* / topological sort all take the full adjacency. No `Graph.AddEdge(u,v,w)` incremental shortest-path / streaming triangle counter (review 281 = temporal-graphs, review 224-ST27 = streaming triangles) |
| info/lz, info/mdl | yes | no | no | minor | `LempelZivComplexity(symbols)` whole-sequence; `RollingComplexity` is window-batch not incremental; no online LZ-symbol counter |
| infogeo | yes | no | no | none | Bregman + f-divergences are pointwise / pairwise functions |
| linalg | yes | no | no | **YES** | All decompositions (LU, QR, Cholesky, PCA) take the full matrix. No incremental SVD / `OnlinePCA.Update(row)` / Frequent-Directions sketch (review 261 owns this gap) |
| optim | yes | partial | no | medium | gradient descent / L-BFGS / FBS / ADMM all run their own loops to convergence; consumer cannot drive one step. Stochastic SGD/Adam (review 220-F) absent |
| optim/transport | yes | no | no | minor | Sinkhorn `(P, log_u, log_v) ← Iterate(...)` runs full schedule; no per-iteration step API |
| orbital | yes | no | no | none | Kepler / vis-viva / Hohmann are closed-form; trajectory propagation defers to `chaos.RK4Step` |
| physics | yes | partial | no | minor | `HeatEquation1DStep(u, dt, dx, alpha, out)` at `physics/thermo.go:82` IS streaming (caller drives the loop); other thermo functions are algebraic |
| prob (root) | yes | partial | partial | **YES** | `EMA(prev, new, alpha)` at `prob/jeffreys.go:149` + `BayesianUpdate(prior, lr)` at `prob/prob.go:85` are streaming. But: NO Welford in `prob/` (lives only in `audio/`!), NO running variance, NO Knuth covariance, NO t-digest / KLL / GK quantile sketch, NO HyperLogLog, NO reservoir sampling (review 224 = the canonical 23-primitive gap) |
| prob/copula | yes | no | no | minor | D-Vine `LogPDF` / `HFunctionPass` whole-vector; no per-observation update of vine state |
| prob/conformal | yes | partial | no | minor | `AdaptiveQuantile(scores, alpha, halfLife)` at `adaptive.go:56` re-sorts on every call (O(n log n) per query); no Gibbs-Candes online-conformal stepper (`α_{t+1} = α_t + γ(target - hit_t)`) |
| queue | yes | no | no | maybe | M/M/1, Erlang B/C are closed-form steady-state; no event-driven discrete-event simulator stepper |
| sequence | yes | no | no | minor | edit distance / Sørensen-Dice / Soundex over fixed strings; no streaming Rabin-Karp / suffix-automaton |
| signal | yes | no | no | **YES** (highest-leverage) | `FFT / IFFT / Convolve / MovingAverage / ExponentialMovingAverage / MedianFilter` all ingest the whole signal slice. `audio/spectrogram.Compute` is hop-batched but allocates the whole T×N output up front. Pistachio at 60 FPS wants `STFTState.Push(frame) []complex128` overlap-add / overlap-save. Review 134 (signal-api) and 132 (signal-missing) both flag this. |
| testutil | n/a | n/a | n/a | n/a | testing |
| timeseries/dcc | yes | **both** | **both** | none | `Params.Update(z, Q, qOut)` (per-step at `dcc.go:86`) AND `Params.FilterSeries(zSeries, n, rSeries)` (batch at `:133`); textbook example with explicit doc note "individual single-step Updates are useful for streaming applications" |
| timeseries/garch | yes | no | no | medium | `Model.Filter(eps, sigma2, z)` at `garch.go:55` IS a per-step recurrence internally but the API exposes only the whole-series slice. No `GarchState.Step(eps_t) (sigma2, z)`. Forecast is batch. |
| topology/persistent | yes | no | no | medium | `VietorisRipsComplex` rebuilds from scratch; no incremental persistence (Cohen-Steiner-Edelsbrunner-Morozov 2006); review 286 (Reeb) and 287 (Mapper) flag this |
| zkmark | n/a | n/a | n/a | n/a | crypto-protocol |

**Tally:** 6 packages with first-class streaming (audio, autodiff, changepoint, chaos, control, crypto/rng, plus partial: physics, prob, timeseries/dcc, timeseries/garch). The remaining ~16 are slice-eaters.

## Streaming idioms already in repo (worth preserving)

1. **`(state-struct, Update, Merge, Query)` quadruple** — `audio/fingerprint.go` (Welford + parallel merge) is the canonical mergeable-summary pattern; review 224-ST1 promotes this to `Sketch[T]` interface.
2. **`Step` per-time + `Solve` whole-trajectory pair** — `chaos.RK4Step` vs `chaos.SolveODE` is the cleanest pattern in the repo. `timeseries/dcc.Update` vs `dcc.FilterSeries` follows the same shape and even calls it out in the docs.
3. **Pre-allocated output slice** — `RK4Step(..., out []float64)`, `HeatEquation1DStep(..., out []float64)`, `UpdateFingerprint(fp, x)` all mutate in place. This is the right contract for a 60 FPS hot loop.
4. **Stateful struct with `Reset`** — `PIDController.Update / Reset` at `control/pid.go:77,119` is the canonical "drop-in real-time controller" shape.

## High-value gaps (in priority order)

1. **`signal.STFTState` streaming STFT (overlap-add / overlap-save)** — Pistachio at 60 FPS wants `state.Push(frame) []complex128`. Currently `audio/spectrogram.Compute(samples, frameSize, hopSize, window) [][]complex128` allocates `T×frameSize` complex matrix up front. Review 318 (resampling) and 134 (signal-api) note this. **Single-PR unlock for every realtime-audio consumer.**
2. **`prob.RunningStat` Welford / co-variance** — Welford lives ONLY in `audio/`. Promote to `prob/` (or new `streaming/`) and re-export-shim from audio. Adds: running mean, running variance, running covariance matrix (vector Welford), running skewness/kurtosis (Pébaÿ 2008). ~120 LOC.
3. **`graph.IncrementalDijkstra` / `graph.AddEdge`** — current shortest-path API rebuilds. For Pulse / Sentinel monitoring streams, an incremental SSSP would be load-bearing. Review 281 (temporal graphs) is the umbrella.
4. **`linalg.OnlineSVD` / Frequent-Directions** — covered by review 261 (online-svd) and 224-ST25; the deterministic Liberty-2013 sketch is the clean entry point at ~150 LOC.
5. **`streaming/` package per review 224** — 23 sketches (CMS, MisraGries, HLL, t-digest, KLL, Bloom, Cuckoo, ReservoirR, FrequentDirections, etc.) — would saturate the entire sublinear-space-aggregate axis. Currently zero sketches in repo.
6. **`signal.IIRFilter / FIRFilter` stateful** — `MovingAverage(signal, ws, out)` and `ExponentialMovingAverage(signal, alpha, out)` are batch even though they trivially reduce to per-sample state machines. Wrap as `EMAState{prev float64; alpha float64}` + `Push(x) float64` + `Reset`. ~60 LOC for both.
7. **`optim.SGDState`, `optim.AdamState`** — review 220 (stochastic-opt). Current optim runs its own loop to convergence, so consumers cannot interleave gradient steps with simulation/data-arrival.
8. **`crypto.HashState`** — `Hash(data []byte)` is one-shot. A `HashState.Update(chunk []byte)` + `Sum()` API matches `hash.Hash` from stdlib and unblocks streaming integrity checks (zkmark, conduit telemetry).
9. **`timeseries/garch.GarchState.Step(eps_t)`** — the recurrence is already per-step internally; expose it. ~30 LOC.
10. **`topology/persistent.IncrementalFiltration`** — recompute-from-scratch dominates VR cost; Cohen-Steiner-Edelsbrunner-Morozov 2006 incremental persistence collapses it. Review 286/287/289 territory.

## Recommendations

1. **Adopt the `(StateStruct, Update / Push, Reset, Merge, Query)` quintuple as the universal streaming contract** — it is already proven in `audio/fingerprint.go` (Welford+Merge), `audio/degradation.go` (Welford+window+Reset), `control/pid.go` (PID+Reset), `changepoint/bocpd.go` (Bayesian online), `timeseries/dcc/dcc.go` (per-step+batch). Document this as the official streaming idiom in CLAUDE.md alongside the "out-buffer" allocation rule.
2. **Promote Welford to `prob/`** — three packages (`audio/fingerprint`, `audio/degradation`, the absent prob version) want it. Single source of truth. Re-export shim from `audio/` to avoid breaking consumers.
3. **Twin every per-step API with a slice-driven batch wrapper** — `chaos.RK4Step` + `chaos.SolveODE` is the model. `timeseries/dcc` already does this and even documents the policy. `changepoint/bocpd` does NOT have a batch wrapper but should (`Run([]float64) [][]float64` would be 6 lines).
4. **Reverse: every batch API that internally loops over a recurrence should expose the recurrence** — `garch.Filter` (already a recurrence inside), `signal.MovingAverage`, `signal.ExponentialMovingAverage`, `signal.MedianFilter`, `optim/transport.Sinkhorn`. Each is a 30-LOC unlock for streaming consumers.
5. **Keep batch as the default convenience** — most numerical algorithms (FFT, decompositions, integrators) are easier to use as one-shot calls. Streaming is the *additional* surface, not the replacement. The repo's current batch-default is correct; the gap is the missing additional surface.
6. **`streaming/` package is the right home for sketches** (per review 224) — peer to `compression`/`crypto`, not nested under `prob`. Sketches are dimension-free + language-agnostic + map-reducible — same contract as compression codecs.
7. **Sketch policy:** every new probability/statistics primitive added to `prob/` should ship with a streaming counterpart whenever the math admits one (Welford for moments; t-digest/KLL for quantiles; reservoir for sampling; CMS/HLL for counts). Document this as a Block-F design rule alongside the existing "every function has golden-file vectors" rule.

## Sources

- `C:\limitless\foundation\reality\audio\fingerprint.go:24,59,85,129,173,207` — Welford + Mahalanobis + Merge (Chan-Golub-LeVeque)
- `C:\limitless\foundation\reality\audio\degradation.go:28,57,71,102,183,194` — DegradationTracker streaming Welford + window
- `C:\limitless\foundation\reality\changepoint\bocpd.go:175,310,401` — `Update / RunLengthPosterior / Step` online Bayesian
- `C:\limitless\foundation\reality\chaos\ode.go:36,80,100` — `RK4Step / EulerStep / SolveODE` step-vs-trajectory pattern
- `C:\limitless\foundation\reality\control\pid.go:77,119` — PID `Update / Reset` controller idiom
- `C:\limitless\foundation\reality\timeseries\dcc\dcc.go:86,133` — `Update` per-step + `FilterSeries` batch with explicit "streaming applications" doc note
- `C:\limitless\foundation\reality\timeseries\garch\garch.go:55,116,147` — `Filter / ForecastVariance / Simulate` all batch despite internal per-step recurrence
- `C:\limitless\foundation\reality\signal\fft.go:49`, `signal\filter.go:54,97,130`, `audio\spectrogram\stft.go:53` — batch-only FFT / filters / STFT (Pistachio gap)
- `C:\limitless\foundation\reality\physics\thermo.go:82` — `HeatEquation1DStep` streaming PDE
- `C:\limitless\foundation\reality\crypto\rng.go:42,113,157` — streaming PRNG generators
- `C:\limitless\foundation\reality\autodiff\tape.go:10,68` — Tape-based reverse-mode (forward pass is online; backward is one-shot)
- `C:\limitless\foundation\reality\prob\jeffreys.go:149`, `prob\prob.go:85,101` — streaming `EMA` and `BayesianUpdate` (the only streaming primitives in `prob/`)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\224-new-streaming.md` — sketch canon, 23 missing primitives, identifies `audio/fingerprint` as the ONLY in-repo streaming primitive prior to this audit
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\261-new-online-svd.md`, `286-new-reeb.md`, `287-new-mapper.md`, `281-new-temporal-graphs.md`, `134-signal-api.md`, `132-signal-missing.md` — corroborating gap reports

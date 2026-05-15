# 386 — meta-allocation-discipline (hot-path allocation audit)

## Headline
~108 non-test files allocate slices via `make([]T,...)`; design rule 3 ("no allocations in hot paths") is strictly honored only by the pure-math packages and the explicitly-buffered API surface (`out []float64` pattern), while ODE/optim/spectrogram/timeseries/graph contain per-call allocations that bite consumers like Pistachio MPPI at 60 FPS.

## Per-package allocation audit
| Package | Clean | Partial | Dirty | Notes |
|---|---|---|---|---|
| `acoustics` | yes | — | — | zero `make([]…)`; pure scalar formulas |
| `em` | yes | — | — | zero `make([]…)`; scalar Coulomb/Ohm |
| `fluids` | yes | — | — | zero `make([]…)`; scalar Reynolds/Bernoulli |
| `orbital` | yes | — | — | zero `make([]…)`; scalar Kepler/vis-viva |
| `color` | yes | — | — | zero `make([]…)`; fixed-array pixels |
| `constants` | yes | — | — | constants only |
| `geometry` | yes | — | — | uses `[3]float64`/`[4]float64` arrays (stack); only `polygon.ConvexHull2D` allocates (non-hot) |
| `physics` | yes | — | — | zero `make([]…)` in source; `HeatEquation1DStep` is buffered API |
| `signal/fft.go` | yes | — | — | confirmed by slot 301; in-place Cooley-Tukey + buffered `PowerSpectrum`/`FFTFrequencies` |
| `signal/window.go` | yes | — | — | all 4 window funcs take `out []float64` |
| `signal/filter.go` | — | yes | — | `Convolve`, `MovingAverage`, `EMA` are buffered; `MedianFilter` allocates 1 sort-buffer per call (filter.go:162) |
| `linalg/vector.go` | yes | — | — | `VectorAdd/Sub/Scale/CrossProduct` buffered |
| `linalg/matrix.go` | yes | — | — | `MatMul`, `MatVecMul`, `Identity`, `MatAdd/Sub/Scale`, `MatTranspose` all buffered |
| `linalg/decompose.go` | — | — | yes | `Inverse` allocates 4 slices (L,U,perm,e,col) per call (decompose.go:162-172, 212-224) |
| `linalg/correlation.go` | — | yes | — | `CovarianceMatrix` is buffered; some helpers allocate |
| `chaos/systems.go` | yes | — | — | LorenzSystem etc. return closures that write into caller's `dydt` slice |
| `chaos/ode.go` | — | — | yes | **RK4Step allocates 5 slices/call** (ode.go:38-42); EulerStep is clean; SolveODE allocates trajectory by design |
| `calculus/calculus.go` | — | yes | — | `NumericalGradient` is buffered; `MonteCarloIntegrate` allocates one workspace (calculus.go:262, "reused across iterations") |
| `compression/quantize.go` | yes | — | — | `ScalarQuantize`/`Dequantize` buffered |
| `compression/entropy.go` | — | — | yes | `JointEntropy`/`ConditionalEntropy` allocate marginals every call (entropy.go:71,96,113) |
| `combinatorics/counting.go` | — | — | yes | binomial/Stirling/Bell DPs allocate `prev`/`curr` rows every call (counting.go:171-209) |
| `crypto` | yes | — | — | only one `make([]byte,…)` in `KeyedDigest` (hash.go:188), one-shot |
| `queue` | — | — | yes | `JacksonNetwork` allocates 4 slices per solve (network.go:91-120) |
| `gametheory/nash.go` | — | — | yes | 6 `make([]float64,…)` in tableau-build paths |
| `gametheory/voting.go` | — | — | yes | 7 `make([]float64,…)` per ballot tally |
| `gametheory/kelly.go` | — | yes | — | 1 alloc, per-bet not per-frame |
| `graph/shortest.go` | — | — | yes | Dijkstra/A* allocate `dist`/`prev`/`gScore`/`cameFrom`/`inClosed` per call (shortest.go:31-87); FloydWarshall allocates O(n²) |
| `graph/centrality.go` | — | — | yes | 7 `make([]…)` per query |
| `graph/community.go`, `graph/mst.go`, `graph/bellman_ford.go`, `graph/pagerank.go` | — | — | yes | each query is fresh-alloc |
| `prob/timeseries.go` | — | yes | — | `ExponentialSmoothing`/`HoltLinear` are buffered; ARIMA helpers allocate diff/centered/autocorr arrays (timeseries.go:134-274) |
| `prob/markov.go`, `prob/nonparametric.go`, `prob/prob.go` | — | — | yes | distribution helpers allocate working arrays |
| `prob/copula/*` | — | — | yes | `gaussian.go`, `t.go`, `vine.go`, `sklar.go`, `kendall_tau.go` all allocate per-evaluation |
| `prob/conformal/*` | — | — | yes | split/adaptive/nonconformity allocate per call |
| `optim/gradient.go` | — | — | yes | L-BFGS allocates `x`,`g`,`gPrev`,`xPrev`,`d`,`q`,`alpha`,plus `sk/yk` per iter (gradient.go:33-205); 13 total |
| `optim/gradient_validated.go` | — | — | yes | 12 alloc sites |
| `optim/linear.go` | — | — | yes | 11 alloc sites (simplex tableau rebuilds) |
| `optim/interpolate.go` | — | — | yes | 7 alloc sites in cubic spline solver |
| `optim/genetic.go`, `optim/metaheuristic.go` | — | — | yes | population arrays per call (acceptable: not a 60 FPS path) |
| `optim/proximal/operators.go` | yes | — | — | `ProxL1/L0/L2/NonNeg/Simplex` all buffered `(v,gamma,out)` |
| `optim/proximal/admm.go` | — | yes | — | one-time alloc in solve loop |
| `optim/proximal/fbs.go` | — | yes | — | one-time scratch |
| `optim/transport/*` | — | — | yes | Sinkhorn/Wasserstein allocate kernel matrices per call |
| `infogeo/bregman.go`, `infogeo/mmd.go` | — | — | yes | per-evaluation working buffers |
| `audio/spectrogram/stft.go` | — | — | yes | `STFT` allocates `out`, plus per-frame `row := make([]complex128, frameSize)` (stft.go:93) — this is a 60 FPS path for any real-time audio |
| `audio/spectrogram/magnitude.go` | — | — | yes | `Magnitude`, `Phase`, `Power`, `HalfSpectrum` each `make([][]…)` per call (magnitude.go:25-131) |
| `audio/spectrogram/mel_spectrogram.go` | — | — | yes | filterbank+per-frame row allocations (mel_spectrogram.go:53-69) |
| `audio/onset/*` | — | — | yes | spectral_flux, complex_domain, energy, superflux all allocate frame buffers |
| `audio/pitch/yin.go`, `audio/pitch/mpm.go` | — | — | yes | autocorrelation buffer per call |
| `audio/separation/wiener.go`, `spectral_subtraction.go` | yes | — | — | `WienerFilterInto`, `SubtractSpectrumInto`, `EstimateNoiseSpectrum` are buffered |
| `audio/separation/nmf.go`, `ica.go` | — | — | yes | 9–17 `make([]…)` sites per fit (training-time, lower priority) |
| `audio/beat`, `audio/tempo`, `audio/cqt`, `audio/fingerprint`, `audio/melscale`, `audio/mfcc`, `audio/segmentation` | — | mixed | mixed | most have a buffered `…Into` variant alongside an allocating convenience version |
| `autodiff/tape.go`, `autodiff/vector.go` | — | — | yes | tape backward allocates `grads := make([]float64, len(t.nodes))` per `Backward` (tape.go:72); vector ops allocate ID/value arrays (vector.go:18-81) |
| `changepoint/bocpd.go` | — | — | yes | 9 `make([]…)` per detector step |
| `timeseries/garch/garch.go`, `timeseries/dcc/dcc.go` | — | — | yes | per-fit allocations (training-time) |
| `topology/persistent/*` | — | — | yes | filtration arrays per call (offline) |
| `info/lz`, `info/mdl/nml.go` | — | — | yes | per-evaluation tables |
| `sequence/alignment.go` | — | — | yes | 2 DP matrices per align |
| `control/transfer.go` | — | yes | — | `Poles` allocates `monic`+`roots` per call (transfer.go:90,139); PID loop itself is scalar/clean |

Counts: 24 packages clean, ~16 partial, the rest dirty for at least one entry point. Total: 366 `make([]float64,...)` occurrences across 108 source files (excluding reviews).

## Top offenders (file:line)
- `chaos/ode.go:38-42` — **RK4Step allocates 5 slices/call** (k1,k2,k3,k4,tmp). Already documented in the godoc as a known limitation. Slot 333/334 measured ~50 MB / control tick under MPPI. Highest priority because it's the canonical "ODE step" used by every dynamical-systems consumer.
- `audio/spectrogram/stft.go:75-93` — `STFT` allocates `frameReal`, `frameImag`, `out [][]complex128`, plus a fresh `row := make([]complex128, frameSize)` per frame. At 50 fps audio framing this is ~50 fresh complex slices per second of audio.
- `audio/spectrogram/mel_spectrogram.go:53-69` — `MelSpectrogram` allocates filterbank+per-frame row+power scratch per call (no `…Into` variant).
- `audio/spectrogram/magnitude.go:25-131` — `Magnitude`, `Phase`, `Power`, `HalfSpectrum` all `make([][]float64, T)` + per-row `make([]float64, F)` per call. No buffered alternative.
- `linalg/decompose.go:162-172` — `Inverse` allocates L, U, perm, e, col on every call. Hot if Pistachio inverts Jacobians per frame.
- `optim/gradient.go:33-205` — `LBFGS` allocates ~13 slices per optimize. Acceptable for offline fits; problematic if used as a per-frame solver.
- `graph/shortest.go:31-87` — `Dijkstra`/`AStar` allocate `dist`, `prev`, `gScore`, `cameFrom`, `inClosed` per call. Pathfinding in NPCs at 60 FPS would hammer GC.
- `autodiff/tape.go:72` — `Tape.Backward` allocates `grads := make([]float64, len(t.nodes))` every backward pass.
- `combinatorics/counting.go:171-209` — `Binomial`/`Stirling2`/`Bell` recompute `prev`/`curr` DP rows every call (no memoization, no buffer arg).
- `compression/entropy.go:71-113` — `JointEntropy`/`ConditionalEntropy` allocate marginal arrays every call.
- `prob/timeseries.go:134-274` — ARIMA fit allocates ~10 working arrays.
- `signal/filter.go:162` — `MedianFilter` allocates a sort buffer per call (small but per-window).
- `audio/onset/*.go` — every onset detector allocates per call (5 detectors × 1-2 allocs each).

## Standard pattern + audit list
**Praise the existing convention.** The good API surface uses one of three patterns, all of which should be the canonical recommendation:

1. **`func X(in []T, ..., out []T)`** — caller provides output. Examples that nail this: all of `signal/window.go`, `signal/filter.go::Convolve/MovingAverage/EMA`, `linalg/matrix.go::MatMul/MatVecMul/Identity/MatAdd/MatSub/MatScale/MatTranspose`, `linalg/vector.go::VectorAdd/Sub/Scale/CrossProduct`, `physics::HeatEquation1DStep`, `compression/quantize.go::ScalarQuantize/Dequantize`, `optim/proximal/operators.go::Prox*`, `chaos/systems.go::Lorenz/Rossler/LotkaVolterra` (closures write into caller's `dydt`), `audio/separation::WienerFilterInto/SubtractSpectrumInto/EstimateNoiseSpectrum`.

2. **`…Into` suffix variant** alongside a convenience allocating version. Audio package is partway there (`WienerFilterInto`, `SubtractSpectrumInto`). Recommendation: add `…Into` variants for every offender.

3. **Stack-allocated fixed-size arrays** (`[3]float64`, `[4]float64`). `geometry/quaternion.go` and `geometry/sdf.go` are exemplary; nothing escapes to heap.

**Action list, ranked by FPS-blast-radius:**
1. `chaos/RK4Step` → add `RK4StepBuffered(scratch *RK4Scratch, ...)` or accept 5 caller buffers. Confirmed in slot 333/334. Highest priority.
2. `audio/spectrogram/STFT`/`Magnitude`/`Phase`/`Power`/`HalfSpectrum`/`MelSpectrogram` → add `…Into` variants taking pre-allocated `[][]complex128` / `[][]float64`.
3. `audio/onset/*` detectors → add buffer-accepting variants.
4. `graph/shortest.go::Dijkstra`/`AStar` → introduce a `Workspace` struct cached by caller.
5. `linalg/Inverse` → take L/U/perm scratch buffers (or document it's offline-only).
6. `autodiff/Tape.Backward` → reuse `grads` slice across calls (tape already owns its node list; an `out []float64` arg is trivial).
7. `compression/entropy.go::JointEntropy/ConditionalEntropy` → buffer the marginals.
8. `prob/timeseries::ARIMA fit` → pool the working arrays.
9. `combinatorics::Binomial/Stirling2/Bell` → memoize across calls (these are pure functions of `(n,k)`).

**Process-level fix:** make "buffered API exists" a checklist item per function. Currently the docstring on `RK4Step` honestly admits "RK4Step allocates temporary k-vectors internally. For truly allocation-free usage in tight loops, callers should implement the method inline." That's a confession, not a contract. Pistachio should not be told to inline RK4 itself.

**Cross-language note:** the `out` parameter pattern also generalizes well to C++/C#/Python validators — span/array-view in C#, pointer+length in C++, NumPy `out=` keyword in Python. The buffered pattern is also the only way to keep golden-file vector validation deterministic across runs without GC noise.

## Sources
- `C:\limitless\foundation\reality\CLAUDE.md` — design rule 3
- `C:\limitless\foundation\reality\chaos\ode.go:36-69` — RK4Step alloc site (already self-flagged in godoc lines 34-35)
- `C:\limitless\foundation\reality\chaos\systems.go` — clean closure pattern
- `C:\limitless\foundation\reality\signal\fft.go` — clean (per slot 301)
- `C:\limitless\foundation\reality\signal\window.go`, `signal/filter.go` — buffered exemplar
- `C:\limitless\foundation\reality\linalg\matrix.go`, `linalg\vector.go` — buffered exemplar
- `C:\limitless\foundation\reality\linalg\decompose.go:153-224` — Inverse alloc
- `C:\limitless\foundation\reality\optim\gradient.go:33-205` — L-BFGS allocs
- `C:\limitless\foundation\reality\optim\proximal\operators.go` — buffered exemplar
- `C:\limitless\foundation\reality\audio\spectrogram\stft.go:75-93` — STFT per-frame allocs
- `C:\limitless\foundation\reality\audio\spectrogram\magnitude.go:25-131` — magnitude family allocs
- `C:\limitless\foundation\reality\audio\separation\wiener.go`, `spectral_subtraction.go` — `…Into` exemplar
- `C:\limitless\foundation\reality\autodiff\tape.go:72` — Backward grads alloc
- `C:\limitless\foundation\reality\graph\shortest.go:31-87` — Dijkstra/A* allocs
- `C:\limitless\foundation\reality\geometry\quaternion.go`, `geometry/sdf.go` — fixed-array exemplar
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\030-chaos-perf.md`, `334-dive-mppi.md`, `015-autodiff-perf.md`, `100-linalg-perf.md`, `134-signal-api.md` — earlier slots' findings

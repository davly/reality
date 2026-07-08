# Reality

Universal truth encoded in code. Pure math, physics, and constants with zero external dependencies.

## Overview

Reality is the foundational math and science library for the Limitless ecosystem. It provides deterministic, pure functions validated against language-neutral golden-file test vectors (JSON).

**Version:** v0.10.0 (cohort-additive — see CONTEXT.md §11 for pre-2026-07 history; this file is the current inventory)
**Module:** `github.com/davly/reality`
**License:** Apache 2.0
**Go version:** 1.24+
**External dependencies:** None (only Go stdlib)
**Packages:** 70 importable (41 top-level + 29 sub-packages under audio/finance/info/optim/prob/timeseries/topology) — via `GO111MODULE=on go list ./...`, excluding the repo-root package (`honesty_test.go` only, no importable source)
**Public functions:** 797 exported — via `git ls-files '*.go' | grep -v '_test\.go' | xargs grep -hE '^func ([A-Z][A-Za-z0-9_]*(\[[^]]*\])?\(|\([a-zA-Z0-9_]+ \*?[A-Za-z0-9_.\[\]]+\) [A-Z][A-Za-z0-9_]*\()' | wc -l`
**Tests:** 3,008 top-level `--- PASS` (4,372 invocations including subtests; all passing under Go 1.24, zero failures)
**Golden-file fixtures:** 138 JSON files under `testdata/` — via `find . -name "*.json" -path "*testdata*" | wc -l`

> **Note on package count:** The v1.0 design target was 22 core domain packages. Reality has shipped additively past that target through Sessions 22-26, the S55-S60 cohort work (audio cohort, autodiff, infogeo, copula/conformal, timeseries, topology, zkmark, forge, info/{lz,mdl}), and the 2026-06/07 Wave 2 (`w2-reality-*`) and Wave 3/4 (`w34-*`) landings (causal, evidence, fairness, finance/taxlot, forge, moments, optim/{hrp,portfolio}, prob/{evt,hmm,risk}, reliability, retrymath, setsim, slo, spc, timeseries, timeseries/statespace). CONTEXT.md §11 is the historical (Session-25-frozen) inventory; this file is regenerated directly from the repo. The table below lists all 70 packages as of 2026-07-05.

## Packages (70)

### Core math and structure

| Package | Description |
|---------|-------------|
| `linalg` | Linear algebra: vectors, matrices, LU/QR/Cholesky decomposition, PCA, sparse matrices |
| `calculus` | Numerical differentiation and integration: Simpson, trapezoidal, RK4, root finding |
| `prob` | Probability and statistics: distributions, Bayesian inference, hypothesis tests, info theory |
| `prob/conformal` | Conformal prediction (regulator-grade calibration; S55 L01 trio) |
| `prob/copula` | Copula models (Gaussian, t, Archimedean — Clayton + Gumbel; S55 L13 trio) |
| `crypto` | Number theory and cryptographic primitives: primality, modular arithmetic, PRNGs, hashing (canonical ecosystem FNV-1a) |
| `geometry` | Computational geometry: quaternions, SDF primitives, curves, convex hull, projective |
| `signal` | Signal processing: FFT/IFFT, convolution, filters, window functions, Hilbert transform |
| `graph` | Graph algorithms: Dijkstra, A*, topological sort, BFS/DFS, network analysis |
| `constants` | Mathematical, physical, and unit conversion constants (SI 2019, NIST CODATA 2018) |

### Applied physics and engineering

| Package | Description |
|---------|-------------|
| `physics` | Classical mechanics, thermodynamics, material properties, stress/strain |
| `em` | Electromagnetism: Coulomb force, electric field, Ohm's law, RC/LC circuits, resonance |
| `fluids` | Fluid mechanics: Reynolds, Bernoulli, Darcy-Weisbach, drag, lift, terminal velocity |
| `acoustics` | Sound and wave propagation: speed of sound, dB SPL, Sabine RT60, Doppler, A-weighting |
| `orbital` | Astrodynamics: Kepler orbits, vis-viva, Hohmann transfer, escape velocity, Hill sphere |
| `chaos` | Dynamical systems: ODE solvers, Lorenz attractor, Van der Pol, Lyapunov exponents |
| `control` | Classical control theory: PID controllers, transfer functions, Bode analysis, stability |

### Decision, games, queues, optimisation

| Package | Description |
|---------|-------------|
| `gametheory` | Nash equilibrium, Shapley value, minimax, replicator dynamics |
| `trust` | Subjective-logic opinions (belief/disbelief/uncertainty) + fusion + discounting; Dempster-Shafer with explicit conflict K (Jøsang 2016 / Zadeh 1984 goldens) |
| `queue` | Queueing theory: M/M/1, M/M/c, M/G/1, Little's law, Erlang B/C |
| `optim` | Optimisation: bisection, Newton, L-BFGS, simulated annealing, GA, simplex |
| `optim/proximal` | Proximal-operator methods (LASSO closed-form witness; first consumer) |
| `optim/transport` | Optimal transport (Sinkhorn, Wasserstein) |
| `combinatorics` | Permutations, combinations, Catalan, Stirling, Bell, integer partitions |
| `compression` | Lossless/lossy compression primitives: entropy, RLE, delta, Huffman, LZ77 |
| `color` | Color science: 8 color spaces, CIEDE2000, WCAG contrast, Bradford adaptation, blackbody |

### Audio cohort (S56 — Pigeonhole / Howler / Dipstick substrate)

| Package | Description |
|---------|-------------|
| `audio` | Mel filterbank (Slaney 1998), MFCC (DCT-II), Welford fingerprint, DegradationTracker |
| `audio/beat` | Beat tracking |
| `audio/cqt` | Constant-Q transform |
| `audio/onset` | Onset detection (energy, spectral flux, complex domain, SuperFlux) |
| `audio/pitch` | Pitch estimators (autocorrelation, YIN, McLeod, subharmonic summation) |
| `audio/segmentation` | Audio event segmentation (VAD, onset-offset, silence-based) |
| `audio/separation` | Multi-source separation (spectral subtraction, Wiener, FastICA, NMF) |
| `audio/spectrogram` | STFT + visualisation (Plasma/Magma/Viridis/Inferno colourmaps) |
| `audio/tempo` | Tempo estimation |
| `audio/vibration` | Mechanical vibration (fundamental, harmonic energy ratio; Dipstick + FW Torque) |

### Sequence, time, info, geometry-of-information

| Package | Description |
|---------|-------------|
| `sequence` | Edit distances (Levenshtein, Jaro-Winkler, NW/SW alignment, n-grams, Soundex) |
| `timeseries/dcc` | Dynamic Conditional Correlation models |
| `timeseries/garch` | GARCH volatility models (with mutually-attesting autodiff gradient parity) |
| `info/lz` | Lempel-Ziv complexity |
| `info/mdl` | Minimum description length |
| `infogeo` | Information geometry (KL gradients, statistical manifolds) |
| `topology/persistent` | Persistent homology / TDA |
| `changepoint` | Change-point detection |
| `autodiff` | Automatic differentiation (forward + reverse mode; mutual-validation backbone) |
| `zkmark` | ZK-Mirror-Mark substrate (cryptographic provenance) |

### Infrastructure

| Package | Description |
|---------|-------------|
| `testutil` | Golden-file test infrastructure for cross-language validation (JSON test vectors) |
| `conduit` | Fire-and-forget HTTP emit shim for ForgeEcosystemEvents (1-in-10000 sampling default) |
| `forge/session40` | Bedrock corpus + canonical situation hashing (Session 40 forge lift) |
| `pkg/canonical` | Canonical encoding helpers |

### Wave 2 + Wave 3/4 additions (18 — 2026-06/07)

Wave 2 (`w2-reality-*` commits) added functions to four already-tabled packages rather than new packages — continuous/fractional Kelly (`optim`), Probabilistic/Deflated Sharpe (`prob`), normalized LZ76 sequence distance (`info/lz`), Cohen/weighted/Fleiss kappa + Krippendorff alpha (`prob/agreement`) — so those rows are unchanged above. The 18 packages below are the new importable packages Wave 2/3/4 added.

| Package | Description |
|---------|-------------|
| `causal` | Back-door ATE estimation from observational data (Pearl adjustment; composes `graph.BackdoorAdjustmentSet`) |
| `evidence` | Decision-neutral evidence-strength scoring from sample-size backing (weakest-link contract) |
| `fairness` | EEOC four-fifths (80%) adverse-impact rule + Wilson score CIs |
| `finance/taxlot` | Statutory tax-lot kernel: IRC §1091 wash-sale apportionment, §1223(3) holding-period tacking, §1222 ST/LT boundary |
| `forge` | Canonical three-way convergence `Decide()` verdict — single source of truth replacing ~56 flagships' drifted private copies |
| `moments` | General streaming Welford mean/variance (scalar + fixed-dim vector) + Chan-Golub-LeVeque parallel merge |
| `optim/hrp` | Hierarchical Risk Parity portfolio construction (Lopez de Prado 2016) |
| `optim/portfolio` | Composed Black-Litterman posterior + mean-variance/continuous-Kelly weight maps (He-Litterman 1999) |
| `prob/evt` | Extreme Value Theory: GEV/GPD, L-moment/PWM/Hill/MLE, peaks-over-threshold, EVT VaR/ES |
| `prob/hmm` | Hidden Markov models: Forward-Backward, Viterbi, Baum-Welch (Rabiner tutorial) |
| `prob/risk` | Convention-arbitrated risk/performance suite: VaR/CVaR/Sortino/drawdown/Calmar/Omega/beta |
| `reliability` | Reliability-block-diagram (RBD) availability composition + Birnbaum importance |
| `retrymath` | Retry-storm load-amplification calculus + stability predicate (composes `reality/queue`) |
| `setsim` | Generic set-similarity coefficients: Jaccard, Sørensen-Dice, overlap (Szymkiewicz-Simpson) |
| `slo` | SRE multiwindow burn-rate alerting algebra (SRE Workbook ch. 5) |
| `spc` | Average-run-length (ARL) calibration for CUSUM/EWMA control charts |
| `timeseries` | Streaming exponentially-weighted mean/variance tracker (`EWMoments`) |
| `timeseries/statespace` | Linear-Gaussian state-space models: Kalman filter, RTS smoother, Durbin-Koopman local-level model |

## Building

```bash
# Verify the module compiles
go build ./...
```

## Testing

```bash
# Run all tests (3,008 top-level PASS)
go test ./...

# Run with verbose output
go test -v ./...

# Run only golden-file tests
go test -run TestGolden ./...

# Run tests for a specific package
go test ./linalg/
go test ./prob/
go test ./physics/
```

## Golden-File Test Vectors

Golden files are JSON documents in `testdata/` that define expected inputs and outputs for every function. They are the language-neutral CONTRACT for these functions: the Go implementation validates against them, and the JSON format is designed so independent Python/C++/C# implementations could validate against the same vectors — though only the Go implementation ships in this repository today.

Format:
```json
{
  "function": "Package.Function",
  "cases": [
    {
      "description": "human-readable description",
      "inputs": {"param": 1.0},
      "expected": 2.0,
      "tolerance": 1e-15
    }
  ]
}
```

Tolerance is per-case, not global. Exact constants use tolerance 0. Iterative algorithms may use wider tolerances. Golden vectors are generated from Go using `math/big` at 256-bit precision, then rounded to `float64`.

## Dependency Position

```
Consumer Apps -> Services -> AI (aicore) -> reality -> math stdlib
```

aicore imports reality. reality imports nothing.

## Design Rules

1. **Zero dependencies.** Only Go standard library.
2. **Golden files are the proof.** Every function has language-neutral test vectors.
3. **Every constant cites its source.** SI 2019, NIST CODATA 2018, or ISO standards.
4. **Pure functions only.** No global state, no goroutines, numbers in / numbers out.
5. **No allocations in hot paths.** Functions accept output buffers. Pistachio calls these at 60 FPS.
6. **Every function cites its mathematical origin.** Three sentences of provenance per function.
7. **Precision documented, not assumed.** Valid input range, numerical precision, and failure modes stated per function.

## Security

Reality is a Tier-0 substrate. See [`SECURITY.md`](./SECURITY.md) for the threat model and vulnerability reporting policy. CI gates: `gosec`, `govulncheck`, and `trivy` (CRITICAL/HIGH) all run with `exit-code: 1`.

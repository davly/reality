# Reality

Universal truth encoded in code. Pure math, physics, and constants with zero external dependencies.

## Overview

Reality is the foundational math and science library for the Limitless ecosystem. It provides deterministic, pure functions validated against cross-language golden-file test vectors (JSON).

**Version:** v0.10.0 (cohort-additive — see CONTEXT.md §11)
**Module:** `github.com/davly/reality`
**License:** Apache 2.0
**Go version:** 1.24+
**External dependencies:** None (only Go stdlib)
**Packages:** 49 importable (32 top-level + audio/info/optim/prob/timeseries/topology sub-packages)
**Public functions:** 584 exported
**Tests:** 2,400+ top-level `--- PASS` (3,322 invocations including subtests; all passing under Go 1.24)
**Golden-file fixtures:** 80 JSON files under `testdata/`

> **Note on package count:** The v1.0 design target was 22 core domain packages. Reality has shipped additively past that target through Sessions 22-26 and the S55-S60 cohort work (audio cohort, autodiff, infogeo, copula/conformal, timeseries, topology, zkmark, forge, info/{lz,mdl}). CONTEXT.md §11 is the canonical inventory of post-Session-25 additions. The table below lists all 49 packages as of 2026-05-15.

## Packages (49)

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

## Building

```bash
# Verify the module compiles
go build ./...
```

## Testing

```bash
# Run all tests (2,400+ top-level)
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

Golden files are JSON documents in `testdata/` that define expected inputs and outputs for every function. The same JSON files are used by Go, Python, C++, and C# implementations to ensure cross-language consistency.

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
2. **Golden files are the proof.** Every function has cross-language test vectors.
3. **Every constant cites its source.** SI 2019, NIST CODATA 2018, or ISO standards.
4. **Pure functions only.** No global state, no goroutines, numbers in / numbers out.
5. **No allocations in hot paths.** Functions accept output buffers. Pistachio calls these at 60 FPS.
6. **Every function cites its mathematical origin.** Three sentences of provenance per function.
7. **Precision documented, not assumed.** Valid input range, numerical precision, and failure modes stated per function.

## Security

Reality is a Tier-0 substrate. See [`SECURITY.md`](./SECURITY.md) for the threat model and vulnerability reporting policy. CI gates: `gosec`, `govulncheck`, and `trivy` (CRITICAL/HIGH) all run with `exit-code: 1`.

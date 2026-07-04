# Reality

Universal truth encoded in code. Pure math, physics, constants. Zero dependencies. Apache 2.0 open source.

## Quick Reference

- **Version:** v0.10.0 (cohort-additive — see CONTEXT.md §11)
- **Go module:** `github.com/davly/reality`
- **License:** Apache 2.0
- **Port:** None (library, not a service)
- **Packages:** 50 importable (32 top-level + audio/info/optim/prob/timeseries/topology sub-packages)
- **Public functions:** 584 exported
- **Tests:** 2,400+ top-level `--- PASS` (3,322 invocations including subtests; all passing, zero failures)
- **Golden fixtures:** 80 JSON files
- **Design doc:** `C:/LimitlessGodfather/architecture/UNIVERSAL_TRUTH_FOUNDATION.md`
- **Review synthesis:** `C:/LimitlessGodfather/reviews/reality-review/SYNTHESIS.md`
- **Context:** `CONTEXT.md` in this repo (read this for full background) — §11 is the canonical inventory of post-Session-25 additions

## Packages (50)

> **Historical note:** v1.0 design target was 22 packages. Reality has shipped additively past that target through Sessions 22-26 + S55-S60 cohort work. CONTEXT.md §11 is the authoritative inventory.

### Core math + applied physics (22 — original v1.0 set)

| Package | Description |
|---------|-------------|
| `acoustics` | Sound and wave propagation: speed of sound, dB SPL, Sabine RT60, Doppler, A-weighting |
| `calculus` | Numerical differentiation and integration: Simpson, trapezoidal, RK4, root finding |
| `chaos` | Dynamical systems: ODE solvers, Lorenz attractor, Van der Pol, Lyapunov exponents |
| `color` | Color science: 8 color spaces, CIEDE2000 perceptual distance, WCAG contrast, Bradford adaptation |
| `combinatorics` | Classical combinatorics: permutations, combinations, Catalan, Stirling, Bell numbers, partitions |
| `compression` | Lossless/lossy compression primitives: entropy, RLE, delta encoding, Huffman, LZ77 |
| `constants` | Mathematical, physical, and unit conversion constants (SI 2019, NIST CODATA 2018) |
| `control` | Classical control theory: PID controllers, transfer functions, Bode analysis, stability margins |
| `crypto` | Number theory and cryptographic primitives: primality, modular arithmetic, PRNGs, hash functions (canonical FNV-1a) |
| `em` | Electromagnetism: Coulomb force, electric field, Ohm's law, RC/LC circuits, series/parallel |
| `fluids` | Classical fluid mechanics: Reynolds, Bernoulli, Darcy-Weisbach, drag, lift, terminal velocity |
| `gametheory` | Classical game theory: Nash equilibrium, Shapley value, minimax, replicator dynamics |
| `geometry` | Computational geometry: quaternions, SDF primitives, curves, convex hull, projective geometry |
| `graph` | Pure graph algorithms: Dijkstra, A*, topological sort, BFS/DFS, network analysis |
| `linalg` | Linear algebra: vectors, matrices, LU/QR/Cholesky decomposition, PCA, sparse matrices |
| `optim` | Optimization: bisection, Newton, L-BFGS, simulated annealing, genetic algorithm, simplex |
| `orbital` | Astrodynamics: Kepler orbits, vis-viva, Hohmann transfer, escape velocity, Hill sphere |
| `physics` | Classical mechanics, thermodynamics, material properties, stress/strain |
| `prob` | Probability and statistics: distributions, Bayesian inference, hypothesis testing, information theory |
| `queue` | Queueing theory: M/M/1, M/M/c, M/G/1, Little's law, Erlang B/C |
| `signal` | Signal processing: FFT/IFFT, convolution, filters, window functions, Hilbert transform |
| `testutil` | Golden-file test infrastructure for cross-language validation (JSON test vectors) |

### Post-Session-25 additions (27 — see CONTEXT.md §11 for provenance)

| Package | Description |
|---------|-------------|
| `sequence` | Edit distances (Levenshtein, Jaro-Winkler, NW/SW alignment), n-grams, Soundex (Session 23) |
| `conduit` | Fire-and-forget HTTP emit shim for ForgeEcosystemEvents (Session 24 Wave 6.A5) |
| `audio` | Mel filterbank (Slaney 1998), MFCC (DCT-II), Welford fingerprint, DegradationTracker (S56) |
| `audio/beat` | Beat tracking (S56 audio cohort) |
| `audio/cqt` | Constant-Q transform (S56 audio cohort) |
| `audio/onset` | Onset detection: energy, spectral flux, complex domain, SuperFlux (S56) |
| `audio/pitch` | Pitch estimators: autocorrelation, YIN, McLeod, subharmonic summation (S56) |
| `audio/segmentation` | Audio event segmentation: VAD, onset-offset, silence-based (S56) |
| `audio/separation` | Multi-source separation: spectral subtraction, Wiener, FastICA, NMF (S56) |
| `audio/spectrogram` | STFT + visualisation (Plasma/Magma/Viridis/Inferno colourmaps) (S56) |
| `audio/tempo` | Tempo estimation (S56 audio cohort) |
| `audio/vibration` | Mechanical vibration: fundamental, harmonic energy ratio (Dipstick + FW Torque) |
| `prob/agreement` | Chance-corrected inter-rater agreement: Cohen kappa, weighted kappa (linear/quadratic), Fleiss kappa, Krippendorff alpha (nominal/ordinal/interval) |
| `prob/conformal` | Conformal prediction (regulator-grade calibration; S55 L01 trio) |
| `prob/numclaim` | Deterministic numeric-claim consistency / invented-value detection: equivalence under rounding + percent<->fraction scaling (portfolio-intelligence number gate) |
| `prob/copula` | Copula models: Gaussian, t, Archimedean — Clayton + Gumbel (S55 L13 trio) |
| `optim/proximal` | Proximal-operator methods (LASSO closed-form witness; first consumer) |
| `optim/transport` | Optimal transport (Sinkhorn, Wasserstein) |
| `timeseries/dcc` | Dynamic Conditional Correlation models |
| `timeseries/garch` | GARCH volatility models (autodiff gradient parity) |
| `info/lz` | Lempel-Ziv complexity |
| `info/mdl` | Minimum description length |
| `infogeo` | Information geometry (KL gradients, statistical manifolds) |
| `topology/persistent` | Persistent homology / TDA |
| `changepoint` | Change-point detection (fresh-start convergence witness) |
| `autodiff` | Automatic differentiation (forward + reverse mode; mutual-validation backbone) |
| `zkmark` | ZK-Mirror-Mark substrate (cryptographic provenance) |
| `forge/session40` | Bedrock corpus + canonical situation hashing (Session 40 forge lift) |
| `pkg/canonical` | Canonical encoding helpers |
| `trust` | Subjective-logic opinions (belief/disbelief/uncertainty/base-rate) + cumulative/averaging fusion + trust discounting; Dempster-Shafer combination with explicit conflict mass K (Yager alternative). Jøsang 2016 / Zadeh 1984 goldens |

## Architecture

One repo. Sub-packages. Single Go module. Go is canonical; Python/C++/C# validate against golden files.

```
reality/
  acoustics/    audio/{beat,cqt,onset,pitch,segmentation,separation,spectrogram,tempo,vibration}/
  autodiff/     calculus/     changepoint/    chaos/        color/
  combinatorics/ compression/ conduit/        constants/    control/
  crypto/       em/           fluids/         forge/session40/
  gametheory/   geometry/     graph/          info/{lz,mdl}/
  infogeo/      linalg/       optim/{proximal,transport}/
  orbital/      physics/      pkg/canonical/  prob/{agreement,conformal,copula}/
  queue/        sequence/     signal/         testutil/     trust/
  timeseries/{dcc,garch}/     topology/persistent/         zkmark/
```

## Dependency Position

```
Consumer Apps -> Services -> AI (aicore) -> reality -> math stdlib
```

aicore imports reality. reality imports nothing.

## Golden-File Testing Infrastructure

The single most important design decision. Every function has golden-file test vectors (JSON) shared across 4 languages (Go, Python, C++, C#).

- Minimum 20 vectors per function, target 30
- Per-function tolerance (not global): exact constants use 0, transcendentals use 1e-11, accumulating ops use 1e-9
- IEEE 754 edge cases mandatory: +Inf, -Inf, NaN, -0.0, subnormals
- Go generates golden files via `math/big` at 256-bit precision; all other languages validate against them

## Key Design Rules

1. **Golden files are the proof.** Every function has golden-file test vectors.
2. **Zero dependencies.** Only the language's standard math library.
3. **No allocations in hot paths.** Functions accept output buffers. Pistachio calls these at 60 FPS.
4. **Every function cites its source.** Mathematical provenance as queryable metadata.
5. **Precision documented, not assumed.** Every function states valid input range, precision, and failure modes.
6. **Reimplement from first principles.** Do not wrap existing libraries.

## Building / Testing

```bash
go test ./...              # Run all tests (2,400+ top-level PASS)
go test -run TestGolden ./...  # Run golden-file validation only
go test -v ./...           # Verbose output
```

## Security

Tier-0 substrate. See [`SECURITY.md`](./SECURITY.md) for threat model + reporting policy. CI gates: `gosec` / `govulncheck` / `trivy` all run with `exit-code: 1`.

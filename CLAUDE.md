# Reality

Universal truth encoded in code. Pure math, physics, constants. Zero dependencies. Apache 2.0 open source.

## Quick Reference

- **Version:** v0.10.0 (cohort-additive — see CONTEXT.md §11 for pre-2026-07 history; this file is the current inventory)
- **Go module:** `github.com/davly/reality`
- **License:** Apache 2.0
- **Port:** None (library, not a service)
- **Packages:** 70 importable (41 top-level + 29 sub-packages under audio/finance/info/optim/prob/timeseries/topology) — derived via `GO111MODULE=on go list ./...` (excludes the repo-root package, which holds only `honesty_test.go` and has no importable source)
- **Public functions:** 797 exported — derived via `git ls-files '*.go' | grep -v '_test\.go' | xargs grep -hE '^func ([A-Z][A-Za-z0-9_]*(\[[^]]*\])?\(|\([a-zA-Z0-9_]+ \*?[A-Za-z0-9_.\[\]]+\) [A-Z][A-Za-z0-9_]*\()' | wc -l`
- **Tests:** 3,008 top-level `--- PASS` (4,372 invocations including subtests via `=== RUN`; all passing, zero failures) — `GO111MODULE=on go test -v ./...`
- **Golden fixtures:** 138 JSON files — `find . -name "*.json" -path "*testdata*" | wc -l`
- **Design doc:** `C:/LimitlessGodfather/architecture/UNIVERSAL_TRUTH_FOUNDATION.md`
- **Review synthesis:** `C:/LimitlessGodfather/reviews/reality-review/SYNTHESIS.md`
- **Context:** `CONTEXT.md` in this repo (read this for full background) — §11 is historical (frozen at Session 25 / 2026-05-01); this file (`CLAUDE.md`) is the current, re-derived inventory as of 2026-07-05

## Packages (70)

> **Historical note:** v1.0 design target was 22 packages. Reality has shipped additively past that target through Sessions 22-26, the S55-S60 cohort work, and the 2026-06/07 Wave 2 and Wave 3/4 (`w2-reality-*` / `w34-*`) landings. Counts on this page are derived directly from the repo (see the Quick Reference commands above), not carried forward from prior doc revisions.

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

### Post-Session-25 additions (30 — see CONTEXT.md §11 for provenance)

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

### Wave 2 + Wave 3/4 additions (18 — 2026-06/07, see git log `w2-reality-*` / `w34-*` commits for provenance)

> `trust` (above) and `prob/numclaim` (above) also landed as part of this cohort (`w34-trust`, `w34-w4-numclaim`) but are tabled with the post-Session-25 set because they were already documented there. Wave 2 (`w2-reality-*`) added functions to four already-documented packages rather than new packages: continuous/fractional Kelly in `optim` (`w2-reality-kelly`), Probabilistic/Deflated Sharpe in `prob` (`w2-reality-dsr`), normalized LZ76 sequence distance in `info/lz` (`w2-reality-ncd`), and Cohen/weighted/Fleiss kappa + Krippendorff alpha in `prob/agreement` (`w2-reality-agreement`) — no new rows needed for those. The 18 packages below are the genuinely new importable packages this cohort added.

| Package | Description |
|---------|-------------|
| `causal` | Back-door average-treatment-effect (ATE) estimation from observational data (Pearl adjustment; composes `graph.BackdoorAdjustmentSet`) |
| `evidence` | Decision-neutral evidence-strength scoring from sample-size backing (+ optional effect magnitude / tier weight); weakest-link contract |
| `fairness` | EEOC four-fifths (80%) adverse-impact rule + Wilson score confidence intervals for selection-process disparate-impact detection |
| `finance/taxlot` | Statutory tax-lot kernel: IRC §1091 proportional wash-sale apportionment + basis adjustment, §1223(3) holding-period tacking, §1222 ST/LT boundary |
| `forge` | Canonical three-way convergence `Decide()` verdict (Uncertain/Converged/Escape) — single source of truth for the 0.65 floor + min-observation count, replacing ~56 flagships' drifted private copies |
| `moments` | General streaming (online, single-pass) Welford mean/variance for scalars and fixed-dimension vectors + Chan-Golub-LeVeque parallel merge |
| `optim/hrp` | Hierarchical Risk Parity portfolio construction (Lopez de Prado 2016): correlation-distance, single-linkage clustering, quasi-diagonalization, recursive bisection |
| `optim/portfolio` | Composed Black-Litterman posterior + mean-variance / continuous-Kelly weight maps over a covariance matrix (He-Litterman 1999) |
| `prob/evt` | Extreme Value Theory: GEV/GPD, L-moment/PWM/Hill/MLE estimators, peaks-over-threshold, EVT Value-at-Risk/Expected-Shortfall and return levels |
| `prob/hmm` | Hidden Markov models: Forward-Backward (log-space scaled), Viterbi decoding, Baum-Welch (EM) re-estimation for categorical emissions (Rabiner tutorial) |
| `prob/risk` | Convention-arbitrated portfolio risk/performance suite: VaR (historical/Gaussian/Cornish-Fisher), CVaR/ES, Sortino, max drawdown, Calmar/Omega/Information ratios, beta |
| `reliability` | Reliability-block-diagram (RBD) availability composition over dependency graphs + Birnbaum importance |
| `retrymath` | Retry-storm load-amplification calculus + stability predicate (backoff schedules, decorrelated jitter); composes with `reality/queue` |
| `setsim` | Generic set-similarity coefficients over slices of comparable elements: Jaccard, Sørensen-Dice, overlap (Szymkiewicz-Simpson) |
| `slo` | SRE multiwindow burn-rate alerting algebra (Google SRE Workbook ch. 5) |
| `spc` | Average-run-length (ARL) calibration for CUSUM/EWMA control charts (Siegmund) |
| `timeseries` | Streaming exponentially-weighted mean/variance tracker (`EWMoments`) — the variance leg of the EWMA control chart (siblings `timeseries/garch`, `timeseries/dcc`) |
| `timeseries/statespace` | Linear-Gaussian state-space models: Kalman filter (1960), Rauch-Tung-Striebel smoother (1965), Durbin-Koopman univariate local-level model (2012) |

## Architecture

One repo. Sub-packages. Single Go module. Go is canonical and is the only implementation shipped in this repo; external Python/C++/C#/TS ports in other repos validate against the same golden files (see Golden-File Testing Infrastructure below).

```
reality/
  acoustics/    audio/{beat,cqt,onset,pitch,segmentation,separation,spectrogram,tempo,vibration}/
  autodiff/     calculus/     causal/         changepoint/    chaos/        color/
  combinatorics/ compression/ conduit/        constants/      control/
  crypto/       em/           evidence/       fairness/       finance/taxlot/
  fluids/       forge/{session40}/            gametheory/     geometry/
  graph/        info/{lz,mdl}/ infogeo/       linalg/         moments/
  optim/{hrp,portfolio,proximal,transport}/   orbital/        physics/
  pkg/canonical/ prob/{agreement,conformal,copula,evt,hmm,numclaim,risk}/
  queue/        reliability/  retrymath/      sequence/       setsim/
  signal/       slo/          spc/            testutil/       trust/
  timeseries/{dcc,garch,statespace}/          topology/persistent/  zkmark/
```

## Dependency Position

```
Consumer Apps -> Services -> AI (aicore) -> reality -> math stdlib
```

aicore imports reality. reality imports nothing.

## Golden-File Testing Infrastructure

The single most important design decision. Every function has golden-file test vectors (JSON) that are a language-neutral CONTRACT — Go generates and validates against them canonically; no non-Go implementation ships inside this repo (see `honesty_test.go`'s `TestNoUnbackedCrossLanguageClaim`, which fails the build if this file ever over-claims that). Real external ports that DO consume this contract live in their own repos: `flagships/pistachio/engine/foundation/reality/` (C++, e.g. `optim/transport/Sinkhorn.cpp`), `sdk/limitless-dotnet/Limitless.Math/` (C#, e.g. `GameTheory.cs`, `LinearAlgebra.cs`, `Probability.cs`), `flagships/horizon/src/horizon/foundation/reality/` and `flagships/relic` (Python, with their own `golden_test.py`), plus newer TypeScript ports in `apps/grammarfix`, `apps/deadlinedocs`, `apps/agmkit`, and `apps/wisecraft` (each with `*.conformance.test.ts` / port-specific test files under `src/lib/`).

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
go test ./...              # Run all tests (3,008 top-level PASS)
go test -run TestGolden ./...  # Run golden-file validation only
go test -v ./...           # Verbose output
```

## Security

Tier-0 substrate. See [`SECURITY.md`](./SECURITY.md) for threat model + reporting policy. CI gates: `gosec` / `govulncheck` / `trivy` all run with `exit-code: 1`.

# Reality

Universal truth encoded in code. Pure math, physics, constants. Zero dependencies. MIT open source.

## Quick Reference

- **Version:** v0.10.0
- **Go module:** `github.com/davly/reality`
- **License:** MIT
- **Port:** None (library, not a service)
- **Tests:** 1,965 (22 packages, all passing, zero failures)
- **Design doc:** `C:/LimitlessGodfather/architecture/UNIVERSAL_TRUTH_FOUNDATION.md`
- **Review synthesis:** `C:/LimitlessGodfather/reviews/reality-review/SYNTHESIS.md`
- **Context:** `CONTEXT.md` in this repo (read this for full background)

## Packages (22)

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
| `crypto` | Number theory and cryptographic primitives: primality, modular arithmetic, PRNGs, hash functions |
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

## Architecture

One repo. Sub-packages. Single Go module. Go is canonical; Python/C++/C# validate against golden files.

```
reality/
  acoustics/    calculus/     chaos/        color/
  combinatorics/ compression/ constants/    control/
  crypto/       em/           fluids/       gametheory/
  geometry/     graph/        linalg/       optim/
  orbital/      physics/      prob/         queue/
  signal/       testutil/
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
go test ./...              # Run all tests (1,965)
go test -run TestGolden ./...  # Run golden-file validation only
go test -v ./...           # Verbose output
```

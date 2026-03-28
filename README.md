# Reality

Universal truth encoded in code. Pure math, physics, and constants with zero external dependencies.

## Overview

Reality is the foundational math and science library for the Limitless ecosystem. It provides deterministic, pure functions validated against cross-language golden-file test vectors (JSON).

**Version:** v0.10.0
**Module:** `github.com/davly/reality`
**License:** MIT
**Go version:** 1.24+
**External dependencies:** None
**Tests:** 1,965 across 22 packages (all passing)

## Packages (22)

| Package | Description |
|---------|-------------|
| `acoustics` | Sound and wave propagation: speed of sound, dB SPL, Sabine RT60, Doppler, A-weighting |
| `calculus` | Numerical differentiation and integration: Simpson, trapezoidal, RK4, root finding |
| `chaos` | Dynamical systems: ODE solvers, Lorenz attractor, Van der Pol, Lyapunov exponents |
| `color` | Color science: 8 color spaces, CIEDE2000, WCAG contrast, Bradford adaptation, blackbody |
| `combinatorics` | Classical combinatorics: permutations, combinations, Catalan, Stirling, Bell, partitions |
| `compression` | Lossless/lossy compression primitives: entropy, RLE, delta, Huffman, LZ77 |
| `constants` | Mathematical, physical, and unit conversion constants (SI 2019, NIST CODATA 2018) |
| `control` | Classical control theory: PID controllers, transfer functions, Bode analysis, stability |
| `crypto` | Number theory and cryptographic primitives: primality, modular arithmetic, PRNGs, hashing |
| `em` | Electromagnetism: Coulomb force, electric field, Ohm's law, RC/LC circuits, resonance |
| `fluids` | Fluid mechanics: Reynolds, Bernoulli, Darcy-Weisbach, drag, lift, terminal velocity |
| `gametheory` | Game theory: Nash equilibrium, Shapley value, minimax, replicator dynamics |
| `geometry` | Computational geometry: quaternions, SDF primitives, curves, convex hull, projective |
| `graph` | Graph algorithms: Dijkstra, A*, topological sort, BFS/DFS, network analysis |
| `linalg` | Linear algebra: vectors, matrices, LU/QR/Cholesky decomposition, PCA, sparse matrices |
| `optim` | Optimization: bisection, Newton, L-BFGS, simulated annealing, genetic algorithm, simplex |
| `orbital` | Astrodynamics: Kepler orbits, vis-viva, Hohmann transfer, escape velocity, Hill sphere |
| `physics` | Classical mechanics, thermodynamics, material properties, stress/strain |
| `prob` | Probability and statistics: distributions, Bayesian inference, hypothesis tests, info theory |
| `queue` | Queueing theory: M/M/1, M/M/c, M/G/1, Little's law, Erlang B/C |
| `signal` | Signal processing: FFT/IFFT, convolution, filters, window functions, Hilbert transform |
| `testutil` | Golden-file test infrastructure for cross-language validation |

## Building

```bash
# Verify the module compiles
go build ./...
```

## Testing

```bash
# Run all tests (1,965)
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

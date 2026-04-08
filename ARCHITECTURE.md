# reality — Architecture

> The Tier 0 foundation. Pure math, zero dependencies, MIT-licensed, the only public repository in the Limitless ecosystem. Everything else depends on this, and this depends on nothing but the Go standard library.

This document is the architectural reference for the `reality` repository. It is written for readers who already know that reality *exists* (see `CONTEXT.md` for the design history, the v1.0 scope decisions, and the philosophical backdrop) and want to understand *how the code is organised and why it looks the way it does*.

`CONTEXT.md` is the historical / design layer. `ARCHITECTURE.md` (this file) is the structural layer. They should be read in that order.

---

## 1. Role in the ecosystem

Reality sits at the bottom of the Limitless dependency DAG. The arrow points one way:

```
Consumer apps ────────▶ Flagships / services ────────▶ aicore ────────▶ reality ────────▶ Go stdlib math
Verdikt                 Echo, Oracle, Phantom,          (AI plumbing)     (this repo)      (math.*,
Paradox                 Parallax, Sentinel,                                                 math/cmplx)
Rift                    Horizon, Pistachio,
Pulse                   RubberDuck, Nexus,
Lighthouse              delve, synthesis,
…                       causal, recall, …
```

Reality imports nothing outside `math`, `math/cmplx`, `math/rand`, `sort`, `errors`, `strings`, `bytes`, `container/heap`, `encoding/json`, `os`, `runtime`, `path/filepath`, `testing`, `context`, `net/http`, `sync/atomic`, `time`, and `strconv`. It never imports `aicore`, `conduit/store`, `limitless-sdk`, or any flagship.

This is enforced by convention, not by tooling. The test is: `grep -rh '^\s*"[a-z][a-z/]*"' --include="*.go"` should only return stdlib paths and paths starting with `github.com/davly/reality/`. Two sub-packages import from siblings: `physics` imports `constants`, and `prob` imports nothing (it has its own internal math helpers).

### Why this layer exists at all

Before reality, math was scattered: `aicore/echomath`, `aicore/oraclemath`, `aicore/causalmath`, `aicore/parallaxmath`, `aicore/calibration`, plus forked copies in every flagship that needed dot products, FFTs, or Jeffreys updates. Brier scores computed by Oracle (Go) and Horizon (Python) could disagree by 1e-14. FNV-1a implementations disagreed across languages (the Session 22 P0 fix). The problem was not any individual implementation — it was the absence of a single canonical source.

Reality solves that with two moves:

1. **One implementation per language, validated against a shared set of golden-file test vectors.** Go is canonical. Python, C++, and C# validate against the same JSON vectors.
2. **MIT license.** The strategic reasoning is documented in `CONTEXT.md` §2: the moat is the AI layer above reality, not reality itself; universal truths should be universally accessible; and being public is a credibility signal that private math libraries cannot earn.

As of Session 25, reality is the only repository in the `davly/` GitHub organisation that is public. Every other ecosystem repository is private.

---

## 2. Package organisation (24 packages)

Reality is a single Go module (`github.com/davly/reality`) containing 24 sub-packages. The choice of "one module, many packages" (Option A in the v1.0 design) was made unanimously by nine review agents — see `CONTEXT.md` §2 for the alternatives that were rejected.

### Core math and structure

| Package | Purpose | Notable types/functions | Depends on |
|---|---|---|---|
| `constants` | Mathematical + physical + unit constants. Six-tier truth taxonomy (`Pi` as `const`, `GravitationalConst` as `var` with uncertainty). | `Pi`, `E`, `Phi`, `Avogadro`, `Planck`, `SpeedOfLight`, `StandardGravity`, `GasConstant`, `StefanBoltzmann`, unit conversion tables. | stdlib `math` |
| `linalg` | Linear algebra: vectors, matrices, decompositions, PCA, sparse. | `CosineSimilarity`, `EncodingDistance`, `DimensionWeightedDistance`, `MatMul`, `MatSub`, `Trace`, `FrobeniusNorm`, LU/QR/Cholesky, eigen, `PearsonCorrelation`, `PCAFit`. | stdlib |
| `calculus` | Numerical differentiation and integration. | `Derivative`, `Simpsons`, `Trapezoidal`, `GaussLegendre`, `RK4`, `Brent`. | stdlib |
| `prob` | Probability distributions, hypothesis tests, Bayesian update, time series, information theory. | `NormalPDF/CDF/Quantile` (Acklam), `BetaPDF`, `Poisson`, `Binomial`, `GammaPDF/CDF`, `BayesianUpdate`, `BrierScore`, `KLDivergence`, `TTest`, `ChiSquared`, `FisherExactTest`, `MannWhitneyU`, `BenjaminiHochberg`, `LinearRegression`, `MarkovChain`. | stdlib |
| `crypto` | Number theory + hash functions + PRNGs. **Holds the canonical ecosystem FNV-1a.** | `FNV1a32`, `FNV1a64`, `MurmurHash3_32`, `MillerRabin`, `ModPow`, `EEAD`, `MersenneTwister`, `XorShift64`. | stdlib |
| `geometry` | Quaternions, SDF primitives, curves (Bezier, B-spline), convex hull, projective geometry. | `QuaternionSlerp`, `SDFSphere`, `SDFBox`, `BezierCurve`, `ConvexHull`. | stdlib |
| `signal` | FFT/IFFT (in-place, zero-alloc), digital filters, window functions, Hilbert transform. | `FFT`, `IFFT`, `FIRFilter`, `IIRFilter`, `HannWindow`, `HammingWindow`, `BlackmanWindow`. | stdlib `math`, `math/cmplx` |
| `graph` | Pure graph algorithms on edge lists and adjacency maps. | `Dijkstra`, `BellmanFord`, `AStar`, `TopologicalSort`, `BFS`, `DFS`, `PageRank`, `KruskalMST`, `Louvain`, `BetweennessCentrality`, `EdmondsKarp` (max flow). | `container/heap` |

### Applied physics and engineering

| Package | Purpose | Notable functions |
|---|---|---|
| `physics` | Classical mechanics, thermodynamics, material properties, optics, stress/strain. | `NewtonSecondLaw`, `ProjectilePosition`, `KineticEnergy`, `HookesLaw`, `VonMisesStress`, `IdealGasLaw`, `BlackbodyRadiance`. Imports `constants`. |
| `em` | Electromagnetism: Coulomb, electric field, Ohm, series/parallel resistors, RC/LC circuits, resonance. | `CoulombForce`, `ElectricField`, `OhmsLaw`, `PowerElectric`, `RCTimeConstant`, `ResonantFrequency`. |
| `fluids` | Reynolds, Bernoulli, Darcy-Weisbach, drag, lift, terminal velocity, Stokes drag. | `ReynoldsNumber`, `BernoulliPressure`, `DarcyWeisbach`, `DragForce`, `TerminalVelocity`. |
| `acoustics` | Speed of sound, dB SPL, Sabine RT60, Doppler, A-weighting, resonance. | `SoundSpeed`, `DecibelSPL`, `SabineRT60`, `DopplerShift`, `AWeighting`. |
| `orbital` | Astrodynamics: Kepler, vis-viva, Hohmann transfer, escape velocity, Hill sphere, synodic period. | `KeplerOrbit`, `VisViva`, `HohmannTransfer`, `EscapeVelocity`, `HillSphere`, `OrbitalPeriod`, `TrueAnomaly`, `KeplerEquation`. |
| `chaos` | Dynamical systems: ODE solvers, Lorenz/SIR/Lotka-Volterra, Lyapunov exponents, bifurcation. | `LorenzRHS`, `SIRModel`, `LotkaVolterra`, `LyapunovExponent`, `BifurcationDiagram`. |
| `control` | Classical control theory: PID, transfer functions, Bode plots, stability margins. | `PIDController`, `TransferFunction`, `BodePlot`, `GainMargin`, `PhaseMargin`. |

### Decision, games, queues, optimisation

| Package | Purpose | Notable functions |
|---|---|---|
| `gametheory` | Nash equilibrium (2×2 and n×m), Shapley value, minimax, replicator dynamics, Gale-Shapley stable matching, multi-armed bandits, Kelly criterion, voting power. | `NashEquilibrium2x2`, `ShapleyValue`, `GaleShapley`, `Minimax`, `KellyFraction`, `BanditUCB`. |
| `queue` | Queueing theory: M/M/1, M/M/c, M/G/1, Erlang B/C, Jackson networks, Little's law. | `MM1`, `MMc`, `MG1`, `ErlangB`, `ErlangC`, `LittlesLaw`, `JacksonNetwork`. |
| `optim` | Optimisation: bisection, Newton, Brent, gradient descent, L-BFGS, simulated annealing, genetic algorithm, simplex method, interpolation. | `Bisection`, `NewtonRoot`, `GradientDescent`, `LBFGS`, `SimulatedAnnealing`, `GeneticAlgorithm`, `SimplexMethod`, `LinearInterp`, `CubicSpline`. |
| `combinatorics` | Permutations, combinations, Catalan, Stirling (first + second kind), Bell, integer partitions. | `Factorial`, `Binomial`, `Catalan`, `Stirling1`, `Stirling2`, `BellNumber`, `IntegerPartitions`. |
| `compression` | Lossless compression primitives: Shannon entropy, run-length encoding, delta, Huffman, LZ77, quantisation. | `ShannonEntropy`, `RunLengthEncode`, `Huffman`, `LZ77Encode`. |
| `color` | Color science: 8 color spaces (sRGB/XYZ/Lab/LCH/HSL/HSV/CMYK/OKLab), CIEDE2000, WCAG contrast, Bradford chromatic adaptation, blackbody colour. | `SRGBToLinear`, `DeltaE2000`, `WCAGContrastRatio`, `BradfordAdapt`, `BlackbodyXYZ`. |

### Sequence, testing, and emit

| Package | Purpose | Notable functions |
|---|---|---|
| `sequence` | **Added Session 23.** Edit distances, sequence alignment, n-grams. Wagner-Fischer Levenshtein (two-row DP), Needleman-Wunsch global alignment, Smith-Waterman local alignment, Hamming, Jaro-Winkler, n-gram extraction. 122 tests. | `LevenshteinDistance`, `JaroWinkler`, `NeedlemanWunsch`, `SmithWaterman`, `HammingDistance`, `NGrams`. |
| `testutil` | Golden-file test infrastructure. Loads JSON fixtures, asserts against per-case tolerance, handles IEEE 754 special values (NaN, ±Inf). Used by every other package's `*_test.go`. | `LoadGolden`, `AssertFloat64`, `AssertFloat64Slice`, `InputFloat64`, `InputFloat64Slice`, `InputInt`, `TestCase`, `GoldenFile`. |
| `conduit` | **Added Session 24 (Wave 6.A5).** Fail-silent, non-blocking HTTP shim that publishes `ForgeEcosystemEvents` to the Conduit bus. Supports both unconditional `Emit()` and sampled `EmitSampled()` (default 1-in-10,000) for hot-path math primitives. Field tags match `store.ForgeLifecycleEvent` in the Conduit repo. **This is the ONE package in reality that is not pure math.** | `Emit(ctx, Event)`, `EmitSampled(ctx, Event)`, `Event` struct, `DefaultURL = "http://localhost:8200/v1/events"`. |

`conduit` is a deliberate architectural exception. Reality was designed to be strictly zero-dependency, and `net/http` is strictly stdlib, but publishing events is not pure math. The compromise is:

- The emit shim is isolated in its own package — callers that want the strict zero-dep guarantee simply do not import `reality/conduit`.
- Reality's own hot paths do not emit. The shim is *present* for the sake of flagships that import `reality/prob.BrierScore`, compute it, and then want to emit a Conduit event recording that computation. They can do that by calling `reality/conduit.Emit` themselves; reality does not call emit on their behalf.
- Emits are fire-and-forget with a 100 ms timeout. If Conduit is down, math primitives are unaffected.

---

## 3. Core design principles

These are not aspirational — they are enforced by code review and by the test suite.

### 3.1 Zero external dependencies

`go.mod` has no `require` clauses beyond the module declaration. Every import is either stdlib or another reality sub-package. This is checked informally by grepping for imports during review.

This constraint is load-bearing: it is the reason reality can credibly claim to be "universal truth". Any external dependency introduces a trust boundary that is not under reality's control.

### 3.2 Pure functions, no global state

No package holds mutable global state. No package starts goroutines at `init` time. No package reads files at import time (golden files are read by tests, not by production code). Functions accept inputs and return outputs — the "numbers in, numbers out" discipline.

The one exception is `conduit`, which holds a package-level `atomic.Uint64` sample counter. This is acknowledged in the package doc as the cost of doing the sampling.

### 3.3 No allocations in hot paths

Matrix operations (`linalg.MatMul`, `linalg.LU`) accept pre-allocated output slices. FFT operates in-place on caller-provided `real` and `imag` slices (`signal.FFT` explicitly documents "zero allocation"). Filter and window functions write into caller buffers. The contract is documented function-by-function.

The reason is `CONTEXT.md` §7: "Pistachio calls these at 60 FPS; RubberDuck calls them on every tick." An allocation per call would compound into GC pressure that dominates frame time.

Allocations that *are* necessary (e.g., `sequence.LevenshteinDistance` needs two O(min(m,n)) row buffers) are documented in the function comment.

### 3.4 Every function cites its mathematical provenance

Every public function has a doc comment with:

1. A one-line summary.
2. The formula (as comment text, not as LaTeX, so it renders in `godoc`).
3. Valid input range.
4. Precision guarantee or failure mode.
5. A historical citation — the original publication, author, and year.

Example from `signal/fft.go`:

```go
// FFT computes the discrete Fourier transform using the Cooley-Tukey radix-2
// algorithm. The transform is computed in-place: real and imag are modified
// directly with zero allocation.
//
// Definition: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*n*k / N)
// Precision: 1e-9 for 1024-point, scales with N*log(N) rounding.
//
// Consumers: Pistachio (audio FFT), RubberDuck (spectral analysis),
// Oracle (time-series frequency decomposition).
```

Provenance is queryable metadata, not buried folklore. The idea (from `CONTEXT.md` §7) is that a consumer can `go doc reality/signal.FFT` and see not just the signature but the historical claim the function is making.

### 3.5 Golden-file test vectors as the cross-language contract

The single most important design decision in reality is the golden-file test infrastructure. Go is canonical — it generates the vectors using `math/big` at 256-bit precision and rounds to `float64`. Python, C++, and C# implementations validate against the same JSON files.

Golden files live under `<package>/testdata/<package>/<function>.json` (package-local, e.g. `linalg/testdata/linalg/lu_decompose.json`) or under top-level `testdata/<package>/<function>.json` for packages that did not vendor their fixtures yet. The format is defined in `testutil/golden.go`:

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

Key design choices:

- **Per-case tolerance, not global.** Exact operations use 0. Transcendentals use 1e-11. Matrix multiply uses 1e-10. FFT uses 1e-9. This is documented per-case, not buried in a constant.
- **`expected` may be scalar or vector.** JSON decodes to `float64` or `[]any`; `testutil.AssertFloat64` and `testutil.AssertFloat64Slice` handle both.
- **IEEE 754 special values are first-class.** NaN matches NaN, +Inf matches +Inf, −Inf matches −Inf. This is handled in `testutil.AssertFloat64`.
- **Mandatory edge cases.** Every golden file must include +Inf/−Inf, NaN, −0.0, subnormals, empty arrays, single-element arrays, all-zero inputs. The minimum per-function target is 20 vectors, with 30 being the goal.

Current fixture count is **73 JSON files** across all packages. Coverage is uneven: `em`, `orbital`, `graph`, `linalg`, `prob`, `fluids`, and `acoustics` are well covered; `signal`, `combinatorics`, `chaos`, `compression`, and `color` have 1-2 fixtures each. Expanding this is a known Session 25+ expansion axis.

### 3.6 Deterministic, platform-independent output

No package uses `time.Now()`, `os.Hostname()`, or any non-deterministic source (except `crypto/rng.go` which is explicitly a PRNG and takes a seed). No package reads environment variables (except `conduit/emit.go` which reads `CONDUIT_URL` for endpoint override and `REALITY_CONDUIT_SAMPLE` for sample rate). No package uses `math/rand.Int()` without an explicit seed.

The determinism guarantee is: given the same inputs, every public function returns byte-identical output across runs, across machines, and across operating systems. The golden files enforce this.

### 3.7 Reimplement from first principles, do not wrap

Reality does not wrap NumPy, gonum, Eigen, or MathNet. Every function is implemented from the mathematical definition. Optional adapter packages may live elsewhere (`limitless-py/reality_numpy.py`, etc.) but the core must stand alone.

The reason: if the core wraps an external library, the "zero dependencies" claim becomes a lie and the cross-language guarantee becomes "whatever the five wrappers happen to agree on". Reimplementation is a one-time cost for a permanent property.

---

## 4. Dependency graph between sub-packages

```
                    constants
                        │
                        ▼
                    physics  ─────► em
                        │
                        │
      linalg ◄──────────┘
        │
        ├───► prob
        ├───► geometry
        ├───► signal
        └───► graph

  crypto (standalone)         sequence (standalone)
  calculus (standalone)       combinatorics (standalone)
  chaos (stdlib only)         compression (stdlib only)
  control (stdlib only)       color (stdlib only)
  fluids (stdlib only)        queue (stdlib only)
  gametheory (stdlib only)    orbital (stdlib only)
  acoustics (stdlib only)     optim (stdlib only)

  testutil (used by every package's tests; not imported by production code)
  conduit (standalone; opt-in emit shim)
```

Key constraints (from `CONTEXT.md` §3):

1. **No cycles.** The graph is strictly acyclic. `physics` imports `constants`, but `constants` does not import anything.
2. **A consumer can import `reality/linalg` alone** without pulling in physics or prob. Go's "compile only what is imported" rule makes this zero-cost.
3. **`testutil` is test-only.** It is imported by `*_test.go` files, never by production code.
4. **`conduit` is opt-in.** No reality package imports `conduit`; flagships that want emit must import it explicitly.

The DAG is simpler than the original v1.0 sketch because most domain packages turned out to need only stdlib. `chaos`, `control`, `fluids`, `gametheory`, `acoustics`, etc. all implement their own math from scratch rather than depending on `linalg` or `calculus`. This is a feature: it lets users pick and choose, and it keeps the blast radius of any single-package refactor small.

---

## 5. Language canonicalisation strategy

Go is canonical. The reasons (from `CONTEXT.md` §2):

1. Go already had the largest math test suite in the ecosystem (1,044 aicore tests before reality extracted them).
2. Go already had the golden-file pattern (FNV-1a vectors, bit-for-bit embedding golden tests) established.
3. Go has `math/big` for 256-bit precision vector generation.
4. The ecosystem's AI layer is Go; Go was the natural source of truth.

The cross-language story:

| Language | Role | Build path |
|---|---|---|
| **Go** (this repo) | Canonical. Generates golden vectors. | `go test ./...` |
| **Python** | `limitless-py/reality.py` — pure-Python core with zero dependencies. Optional `_numpy.py` adapter for speed. Pure path validates against golden files in CI. | Not yet shipped. |
| **C++** | `limitless-cpp/reality.hpp` — header-only, uses `<cmath>`. Optional Eigen bridge header for acceleration. SIMD paths behind `REALITY_SIMD` define. | Not yet shipped. |
| **C#** | `Limitless.AI.Reality` — from-scratch port, uses `System.Math`. Optional `Limitless.AI.Reality.MathNet` adapter assembly. | Not yet shipped. |

As of Session 25, **only the Go canonical implementation has shipped.** The Python, C++, and C# ports are in the "planned" column. This is a known P1 gap — the golden-file infrastructure exists and is exercised by Go tests, but the cross-language guarantee is currently only *designed*, not *demonstrated*.

---

## 6. Public API surface

This section lists the public types and functions that are stable enough to cite. It is not exhaustive — 426 public functions are too many to enumerate here — but it captures the entry points most commonly used by the ecosystem.

### 6.1 `constants`

- `Pi`, `E`, `Phi`, `Sqrt2`, `Sqrt3`, `Ln2`, `Ln10`, `Log2E`, `Log10E`, `EulerGamma` — mathematical constants as `const`.
- `SpeedOfLight`, `Planck`, `PlanckReduced`, `ElementaryCharge`, `VacuumPermittivity`, `VacuumPermeability`, `Boltzmann`, `Avogadro`, `GasConstant`, `StandardGravity`, `GravitationalConst`, `StefanBoltzmann`, `AtmPressure` — physical constants.
- Unit conversion tables (SI 2019, NIST CODATA 2018).

### 6.2 `linalg`

- `CosineSimilarity(a, b []float64) float64`
- `EncodingDistance(a, b []float64) float64`
- `DimensionWeightedDistance(a, b, weights []float64) float64`
- `MatMul(a, b, out []float64, rowsA, colsA, colsB int)`
- `MatSub`, `Trace`, `FrobeniusNorm`
- `LUDecompose(a []float64, n int) (L, U []float64, perm []int, err error)`
- `QRDecompose`, `CholeskyDecompose`
- `Eigen(a []float64, n int) (eigenvalues []float64, eigenvectors []float64)`
- `PCAFit(data [][]float64, k int) (components [][]float64, explained []float64)`
- `PearsonCorrelation(x, y []float64) float64`

### 6.3 `prob`

- Distributions: `NormalPDF/CDF/Quantile`, `BetaPDF/CDF`, `PoissonPMF/CDF`, `BinomialPMF/CDF`, `GammaPDF/CDF`, `ExponentialPDF/CDF`, `UniformPDF/CDF`.
- Tests: `TTest`, `PairedTTest`, `ChiSquaredTest`, `FisherExactTest`, `MannWhitneyU`, `KolmogorovSmirnov`.
- Bayesian: `BayesianUpdate(prior, likelihood)`, `BrierScore(predicted, actual)`, `KLDivergence`.
- Regression: `LinearRegression`, `BenjaminiHochberg(pvalues, q)`.
- Time series: `MarkovChain`, `MarkovSimulate`, `MarkovSteadyState`, `ARIMA` (partial).

### 6.4 `crypto` (canonical ecosystem hashing)

- `FNV1a32(data []byte) uint32`
- `FNV1a64(data []byte) uint64` — **the ecosystem-canonical hash**. Every service that computes a situation hash must delegate to this.
- `MurmurHash3_32`
- `MillerRabin(n *big.Int, iterations int) bool`
- `ModPow`, `ModInverse`, `ExtendedEuclidean`
- `MersenneTwister`, `XorShift64`

### 6.5 `signal`

- `FFT(real, imag []float64)` — in-place, zero-alloc, radix-2 Cooley-Tukey
- `IFFT(real, imag []float64)`
- `FIRFilter(input, coeffs, output []float64)`
- Window functions: `HannWindow`, `HammingWindow`, `BlackmanWindow`, `Kaiser`
- `HilbertTransform`

### 6.6 `graph`

- `AdjacencyList(edges []Edge) map[string][]string`
- `Dijkstra(adj, start string) (dist map[string]float64, prev map[string]string)`
- `AStar`, `BellmanFord`, `TopologicalSort`, `BFS`, `DFS`
- `PageRank(adj, damping, tolerance)`
- `KruskalMST`, `Louvain` (community detection), `BetweennessCentrality`, `EdmondsKarp` (max flow)

### 6.7 `testutil` (test-only)

- `LoadGolden(t *testing.T, path string) GoldenFile` — loads a JSON fixture relative to the caller's source file.
- `AssertFloat64(t, tc, got)` — NaN-aware, Inf-aware, tolerance-aware.
- `AssertFloat64Slice(t, tc, got)` — same semantics, per-element.
- `InputFloat64`, `InputFloat64Slice`, `InputInt` — typed input extractors.

### 6.8 `conduit` (opt-in emit)

- `Emit(ctx context.Context, e Event)` — unconditional, fire-and-forget, 100 ms timeout.
- `EmitSampled(ctx context.Context, e Event)` — one in `SampleRate` (default 10,000).
- `Event` struct with fields matching `store.ForgeLifecycleEvent` in the Conduit repo.

---

## 7. Standards compliance notes

Reality predates the Wave 8.1 synthesis (`reviews/session_24_adversarial/WAVE_8_1_SYNTHESIS.md` in the LimitlessGodfather repo) by many sessions. Wave 8.1 distilled the delve embed corpus into a canonical five-noun skeleton and 10 universal findings; those findings are mostly about *observation services* (Query, Corpus, Step, Result, Escape), and they do not map onto a pure-math library.

The checklist from Wave 8.1 § 4 applied to reality:

| Wave 8.1 finding | Applies to reality? | Notes |
|---|---|---|
| Five-noun skeleton (Query · Corpus · dig/walk · Result · EscapeReason) | **No.** | Reality exposes pure functions, not a walker. There is no corpus and no escape to report — functions return numbers, or NaN, or panic on documented invalid input. |
| FNV-1a 64-bit situation hash over sorted dimensions | **Reality is the source.** | `crypto/hash.go` holds `FNV1a64`. Other services compute situation hashes *by calling this*. Canonical vectors live at `LimitlessGodfather/architecture/fnv1a_canonical_vectors.json`. |
| Three-way verdict (Dominated / Uncertain / Refuses) | **No.** | Reality has no opinion about dominance or refusal. That is a walker concern. (Session 25 open item: a `reality/prob/jeffreys.go` file with `ThreeWayVerdict` primitives would let walkers delegate the math.) |
| Closed escape-reason enum (7–10 variants typical) | **No.** | Reality does not escape. Functions may return NaN, or `(result, error)`, or panic on programmer error (out-of-bounds, non-power-of-two FFT length). |
| Jeffreys (0.5, 0.5) quality-weighted dominance | **Not yet.** | This is the Session 25 open item. The math (Beta prior, quality-weighted update) is primitive enough to belong in `reality/prob`, but currently lives in individual flagships. |
| Fail-silent fire-and-forget Conduit emit with 100 ms timeout | **Yes, via `reality/conduit`.** | `conduit/emit.go` implements exactly this: non-blocking `go func`, 100 ms context timeout, silent on HTTP failure. |
| Deterministic ordering via sorting at collection-return seams | **Yes, implicitly.** | Reality functions that return maps (`graph.Dijkstra`) are documented as order-insensitive; functions that return slices return them in defined order. |
| Constructor-invariant refusal before work begins | **Partial.** | Reality validates inputs before computing: `NormalPDF` returns NaN for `sigma <= 0`; `FFT` panics for non-power-of-two length. The principle is the same; the mechanism differs from a walker's structured refusal. |
| Structured escape (no exceptions, escape as return value) | **Partial.** | Go has no exceptions by design. Reality uses NaN for numerical failure, `(result, error)` for I/O failure, and panic for programmer error. This aligns with the Wave 8.1 finding by construction. |

**Bottom line.** Reality is not a delve implementation and does not pretend to be. It provides the *primitives* (FNV-1a, future Jeffreys helpers, arithmetic) that delve implementations call. Wave 8.1 compliance for reality is about making those primitives available and well-tested, not about reshaping reality's API into a walker skeleton.

---

## 8. Testing strategy

### 8.1 Layered testing

Every package has at least two test files:

1. **`<pkg>_test.go`** — unit tests and golden-file validation. Covers the happy paths and documented edge cases.
2. **`<pkg>_edge_test.go`** (where present) — adversarial tests added in commit `5b63907` (Session 9) that probe IEEE 754 edge cases, numerical cliffs, and boundary conditions that the main test file does not cover.

Some packages have multiple sub-test files corresponding to sub-domains (e.g. `prob/distributions_test.go`, `prob/regression_test.go`, `prob/timeseries_test.go`, `prob/nonparametric_test.go`, `prob/markov_test.go`).

### 8.2 Golden-file tests vs. table-driven tests

Tests in reality come in three shapes:

1. **Golden-file tests** — drive test cases from JSON files loaded by `testutil.LoadGolden`. These are the cross-language contract; they must exist for any function that crosses language boundaries.
2. **Table-driven tests** — Go's idiomatic `for _, tc := range cases { t.Run(tc.name, ...) }` pattern. Used for functions where a JSON vector would be overkill (e.g. integer combinatorics, boolean predicates).
3. **Property tests** — where present, check invariants (e.g. FFT followed by IFFT equals the original within tolerance; Cholesky decomposition reconstructed matches the input).

Total test counts:

- 1,585 top-level `--- PASS` lines (Session 25 measurement).
- 2,409 `=== RUN` entries including subtests.
- 73 golden-file JSON fixtures.
- All packages pass under `go test ./...`.

### 8.3 Running the suite

```bash
# The whole thing
go test ./...

# Verbose output
go test -v ./...

# Golden-file tests only
go test -run TestGolden ./...

# One package
go test ./linalg/
go test ./prob/
go test ./signal/
```

The suite takes under 60 seconds on a modern laptop. There are no long-running tests, no network calls (the `conduit` package has no tests), no filesystem writes.

---

## 9. Build, release, and versioning

### 9.1 Current state

- **Version:** v0.10.0 (declared in `README.md` and `CLAUDE.md`).
- **Branch:** `master`. Not yet renamed to `main` (one of ~70 ecosystem repos in this state).
- **Go module:** `github.com/davly/reality`.
- **Go version required:** 1.24+.
- **External dependencies:** none. `go.mod` contains only the module declaration and Go version.
- **GitHub:** `davly/reality`, public, MIT-licensed by declaration (a `LICENSE` file is not yet committed — a zero-risk P2 action).
- **Build:** `go build ./...` — succeeds with no flags, no build tags, no codegen.
- **CI:** Not yet configured in-repo. The design calls for GitHub Actions covering all four languages at v1.0.

### 9.2 Release plan (from `CONTEXT.md` §9)

Reality is following a five-phase release schedule. Session 25 places it approximately at the end of Phase 2:

- **Phase 0** (Weeks 1–2): Golden-file infrastructure. **Done.**
- **Phase 1** (Weeks 3–6): Extract + reorganise existing math. **Done.** `linalg`, `prob`, `crypto`, `signal`, `graph` all hold extracted-and-cleaned code.
- **Phase 2** (Weeks 7–14): Fill the gaps. **In progress.** All 24 packages exist; per-function coverage varies. Session 25 open items (Jeffreys helpers, expanded golden-file coverage) land here.
- **Phase 3** (Weeks 15–18): Pistachio (C++) and RubberDuck (C#) integration. **Not started.** Python port also not shipped.
- **Phase 4** (Weeks 19–20): Open-source preparation. **Partial.** Repo is already public, MIT declared in docs, but `LICENSE` file is missing, CI is missing, and README could use the provenance-and-citation story.

### 9.3 Versioning

Reality uses semantic versioning. v0.x versions may break API between minor releases. v1.0 will be the open-source launch and the first API stability commitment.

---

## 10. Evolution policy

Changes to reality are governed by the inclusion tests from `CONTEXT.md` §7:

1. **Permanence Test** — will this still be true in a hundred years?
2. **Independence Test** — is this true regardless of who observes it?
3. **Precision Test** — can correctness be verified to arbitrary precision?

The "alien test" is a shorthand: would an alien civilisation independently discover this? If yes, it belongs in reality. If it depends on human culture (music theory beyond harmonics, typography, market microstructure), it does not.

This policy has excluded music theory, typography, ASCII-specific string handling, and domain-specific business logic. It has included CIEDE2000 (because perceptual color distance is a measurement of human cognition, not a convention), WCAG contrast (borderline — included because it is a *function* of physical luminance, not a style preference), and Kelly criterion (because it is a theorem about bankroll growth, not a market convention).

When in doubt, the question is: if we deleted this function tomorrow, could consumers reimplement it from a textbook in an hour? If yes, reality is the right place. If it requires proprietary knowledge or context-dependent calibration, it belongs in a flagship, not in reality.

---

## 11. Cross-references

- **`CONTEXT.md`** (this repo) — the design history, v1.0 decisions, package roster, and Session 25 state update. Read this first.
- **`README.md`** (this repo) — the public-facing description and build instructions.
- **`CLAUDE.md`** (this repo) — session handover / quick reference for future Claude sessions.
- **`architecture/UNIVERSAL_TRUTH_FOUNDATION.md`** (LimitlessGodfather) — the 2,124-line original design document. Decisions noted here trace back to sections in that document.
- **`reviews/reality-review/SYNTHESIS.md`** (LimitlessGodfather) — the 679-line 9-agent review synthesis that confirmed Option A, MIT licensing, and the 6-tier truth taxonomy.
- **`architecture/fnv1a_canonical_vectors.json`** (LimitlessGodfather) — the canonical FNV-1a golden vectors that `reality/crypto` must match.
- **`reviews/FLEETWORKS_CROSS_POLLINATION_PLAN.md`** (LimitlessGodfather) — contains the cross-pollination plan for moving shared math out of flagships into reality.
- **`reviews/session_24_adversarial/WAVE_8_1_SYNTHESIS.md`** (LimitlessGodfather) — canonical walker architecture. Applies to reality only via §7 above.
- **`apps/PORTFOLIO.md`** (LimitlessGodfather) — the 15 pure-reality apps possible without AI (Forge, Abacus, Wavelength, Census, Orrery, Ledger Prime, Girder, Terraform, Lattice, etc.).

---

*Written 2026-04-08 as part of Session 25 per-repo audit. Replaces the auto-generated baseline. The document is expected to evolve as Phase 2 (gap-fill) and Phase 3 (cross-language ports) progress.*

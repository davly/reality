# Reality -- Context Document

> Universal truth encoded in code.

This is the foundational context for any session working on the Reality project. Read this first.

---

## 1. What Reality Is

Reality is the lowest software layer in the Limitless ecosystem. It sits beneath aicore, beneath the SDKs, beneath every service and consumer app. It provides pure mathematical and physical functions that are:

- **Deterministic** -- same input, same output, every time, on every platform
- **Proven** -- actual formulas that humanity has discovered, verified, and relied upon for decades or centuries
- **Eternal** -- this code changes only if our understanding of reality changes
- **Zero-dependency** -- only the language's standard math library; no external packages, no network calls, no randomness unless explicitly seeded
- **Cross-language** -- the same functions, with the same golden-file test vectors, implemented in Go, Python, C++, and C#

Reality is not an AI library. It does not call models, manage tokens, or route requests. The dependency arrow points in one direction only:

```
aicore imports reality
reality imports nothing
```

**License:** MIT. Open source from day one. Universal truths should be universally accessible.

**Go module:** `github.com/davly/reality`

---

## 2. Architecture Decisions (Confirmed)

All decisions below were confirmed unanimously by 9 independent review agents. No dissent was recorded.

### Repository structure: One repo, sub-packages, single Go module (Option A)

Four options were evaluated. Option A was selected unanimously:

| Option | Description | Verdict |
|--------|-------------|---------|
| A. One repo, sub-packages | Single go.mod, many packages | **Selected** |
| B. Separate repos per domain | Independent modules | Rejected: diamond deps, cascade releases |
| C. reality + nature split | Math vs. applied science | Rejected: naming confusion, unnecessary boundary |
| D. One repo, multiple modules | golang.org/x pattern | Rejected: tooling complexity, no compelling benefit |

**Why Option A:**
1. No version matrix. One `require` line. No diamond dependencies.
2. Matches the aicore precedent (12 projects import aicore sub-packages).
3. Lowest maintenance burden. One CI pipeline, one release process.
4. Go encourages it. The standard library is a monolith. The compiler only links imported packages.
5. Laws and techniques ship together. F = ma next to Verlet integration.

### Canonical language: Go

Go generates the golden files. All other languages validate against them. This matches the existing aicore pattern where Go is the source of truth.

### Cross-language strategy: Reimplement from first principles, validate via golden files

Do not wrap existing libraries. Reimplement core operations from scratch in each language. Provide optional adapters (NumPy, Eigen, MathNet) as separate modules validated against the core golden files.

- **Python:** Pure Python core (`_core.py`), optional NumPy acceleration (`_numpy.py`). Golden-file CI runs against pure path.
- **C++:** Header-only (`limitless-cpp`). Optional Eigen bridge header. SIMD paths gated by `REALITY_SIMD` define.
- **C#:** From-scratch `Limitless.AI.Reality`. Optional MathNet adapter assembly.

### Precision: 256-bit golden-file test vectors

Before a function's golden file is committed, a Go tool computes each expected output using `math/big` at 256-bit precision, then rounds to `float64`. Per-function tolerance fields in the golden-file JSON format (not a global constant):

| Operation class | Tolerance |
|----------------|-----------|
| Addition, multiply | 1e-15 |
| Dot product (n <= 100) | 1e-12 |
| Dot product (n > 1000) | 1e-9 |
| sqrt, reciprocal sqrt | 1e-12 |
| sin, cos, tan, exp, log | 1e-11 |
| Matrix multiply (n <= 8) | 1e-10 |
| FFT (1024-point) | 1e-9 |
| Procedural generation | 0.0 (exact) |

### License: MIT

Open source from v1.0. The strategic value is credibility signal, not acquisition funnel. The competitive moat is the AI layer above reality, not reality itself.

---

## 3. The Packages (22)

Reality contains 22 sub-packages. The original design called for 16; six additional domains were added during implementation as the physics scope was decomposed into focused packages and new applied-math areas were identified.

| Package | Domain | Description |
|---------|--------|-------------|
| `reality/linalg` | Linear Algebra | Vectors, matrices, decompositions (LU, QR, Cholesky), PCA, sparse matrices |
| `reality/calculus` | Calculus & Analysis | Differentiation, integration (Simpson, trapezoidal), RK4, root finding |
| `reality/prob` | Probability & Statistics | Distributions (Normal, Beta, Poisson, Binomial), Bayesian inference, hypothesis testing (t-test, chi-squared), information theory |
| `reality/physics` | Classical Physics | Mechanics, thermodynamics, material properties, stress/strain analysis |
| `reality/graph` | Graph Theory | Dijkstra, A*, topological sort, BFS/DFS, network analysis |
| `reality/crypto` | Number Theory & Crypto | Primality, modular arithmetic, PRNGs, hash functions |
| `reality/geometry` | Geometry | Quaternions, SDF primitives, curves, convex hull, projective geometry |
| `reality/signal` | Signal Processing | FFT/IFFT, convolution, filters, window functions, Hilbert transform |
| `reality/constants` | Constants | Mathematical constants, CODATA 2018 physical constants (exact as `const`, measured as `var` with uncertainty), unit conversions |
| `reality/color` | Color Science | 8 color spaces, CIEDE2000 perceptual distance, WCAG contrast, Bradford chromatic adaptation, blackbody |
| `reality/gametheory` | Game Theory | Nash equilibrium, Shapley value, minimax, replicator dynamics |
| `reality/queue` | Queueing Theory | M/M/1, M/M/c, M/G/1, Little's law, Erlang B/C |
| `reality/combinatorics` | Combinatorics | Permutations, combinations, Catalan, Stirling, Bell numbers, integer partitions |
| `reality/optim` | Optimization | Bisection, Newton, L-BFGS, simulated annealing, genetic algorithm, simplex method |
| `reality/compression` | Compression | Entropy, RLE, delta encoding, Huffman, LZ77 |
| `reality/control` | Control Theory | PID controllers, transfer functions, Bode analysis, stability margins |
| `reality/chaos` | Dynamical Systems | ODE solvers, Lorenz attractor, Van der Pol oscillator, Lyapunov exponents |
| `reality/fluids` | Fluid Mechanics | Reynolds number, Bernoulli, Darcy-Weisbach, drag, lift, terminal velocity, Stokes |
| `reality/em` | Electromagnetism | Coulomb force, electric field, Ohm's law, power, RC/LC circuits, resonance |
| `reality/acoustics` | Acoustics | Speed of sound, dB SPL, Sabine RT60, Doppler effect, A-weighting |
| `reality/orbital` | Astrodynamics | Kepler orbits, vis-viva, Hohmann transfer, escape velocity, Hill sphere |
| `reality/testutil` | Test Infrastructure | Golden-file test framework for cross-language validation (JSON vectors) |

### Dependency DAG

```
                linalg
               /  |   \
              /   |    \
        calculus  stats  crypto
          |  \     |
          |   \    |
       physics  geometry  signal
          |
      [chemistry]  (v2.0)
          |
      [biology]    (partial v1.0, full v2.0)
```

Key constraints:
- Everything depends on linalg. Nothing depends on everything.
- No cycles. The graph is strictly acyclic.
- A consumer can import `reality/linalg` alone without pulling in physics.

---

## 4. v1.0 Scope

**~397 functions. ~8,990 golden-file test vectors. 5 phases over ~20 weeks.**

### Function count by domain

| Domain | Functions | Golden Vectors |
|--------|----------:|---------------:|
| Linear algebra | 40 | 1,000 |
| Calculus/analysis | 30 | 900 |
| Probability/statistics | 60 | 1,800 |
| Physics | 50 | 1,500 |
| Graph theory | 25 | 500 |
| Number theory/crypto | 15 | 225 |
| Geometry | 35 | 700 |
| Signal processing | 30 | 600 |
| Constants | 25 | 25 |
| Biology (partial) | 5 | 100 |
| Color | 25 | 500 |
| Game theory | 15 | 300 |
| Decision theory | 10 | 200 |
| Queuing theory | 12 | 240 |
| Geodesic | 8 | 160 |
| Sequence operations | 12 | 240 |
| **Total** | **~397** | **~8,990** |

### Golden-file requirements

Every golden-file batch must include:

**IEEE 754 special values (mandatory):**
- `+Inf`, `-Inf` as inputs where meaningful
- `NaN` as input with defined behavior
- `-0.0` (sign bit matters for atan2)
- Subnormal floats (5e-324 in float64)
- Very large values (1e308, 1e-308)

**Structural edge cases (mandatory):**
- Empty input (zero-length arrays)
- Single-element arrays
- Length-mismatched arrays
- All-zero inputs
- Inputs with repeated identical elements

**Minimum 20 vectors per function, target 30.**

---

## 5. What v1.0 Excludes

Explicitly deferred:

| Exclusion | Reason |
|-----------|--------|
| Chemistry | No new mathematical primitives needed. Pure applied physics. Ships in v2.0. |
| Advanced biology | Wright-Fisher, Hodgkin-Huxley, bioinformatics. Beyond the math. v2.0. |
| Full physics engine | Reality provides equations; Pistachio provides the engine (scene graph, collision detection, GPU dispatch). |
| Unit-aware computation | Type signatures reserve space. Full dimensional analysis engine ships in v1.1. |
| Symbolic algebra | Reality operates on floating-point numbers, not symbolic expressions. |
| Quantum mechanics (beyond constants) | No Schrodinger solvers. Tier 3 (educational value, niche demand). |
| General relativity | No metric tensors, no geodesic equations. Tier 3. |
| Formal proof verification | Machine-checked proofs (Lean 4, Coq) are a dream feature. |
| Interactive explorer | WebAssembly formula playground is a dream feature. |
| REST API | Math-as-a-service is a product built ON reality, not part of it. |
| Music theory | Low consumer count. Harmonic series already in physics/acoustics. Separate library if needed. |
| Typography | Aesthetic engineering, not truth. |
| Market microstructure | Single consumer (RubberDuck). Underlying math already in reality. |
| Cryptographic ciphers | Security-critical. The number theory foundation is in reality/crypto; cipher implementations belong in security-audited libraries. |

---

## 6. The Ecosystem Stack

```
Consumer Apps (Verdikt, Paradox, Rift, Pulse, Lighthouse, ...)
    |
    v
Services (Echo, Parallax, Oracle, Phantom, Sentinel, ...)
    |
    v
AI Layer (aicore, limitless-sdk, nexus-ai)
    |
    v
Universal Truth Foundation (reality)       <-- THIS LAYER
    |
    v
Mathematics (Go stdlib math, hardware FPU)
```

Language-specific dependency chains:

```
Go:       app -> limitless-sdk -> aicore -> reality -> math stdlib
Python:   app -> limitless-py  -> reality.py -> math/numpy
C++:      app -> limitless-cpp -> reality.hpp -> <cmath>
C#:       app -> Limitless.Math -> System.Math
```

---

## 7. Design Philosophy

### David's inspiration

> "When learning to code, I loved building physics simulations. Applying simple,
> well-established formulas led to beautiful natural effects. If simple formulas
> make circles feel alive, what could the most advanced mathematical concepts
> achieve?"

### The alien test

A proposed addition to reality must pass: would an alien civilization independently discover this? If the answer is yes, it belongs. If it depends on human culture, language, or convention, it does not.

More precisely, three inclusion tests:

1. **The Permanence Test:** Will this still be true in a hundred years?
2. **The Independence Test:** Is this true regardless of who observes it?
3. **The Precision Test:** Can correctness be verified to arbitrary precision?

### Provenance: every function cites its mathematical origin

Not as a buried comment -- as queryable metadata. Three sentences of history per function. Who discovered it, when, and why it matters.

```go
// FFT computes the Fast Fourier Transform using the Cooley-Tukey algorithm.
//
// The Fourier transform decomposes a signal into its constituent frequencies.
// Joseph Fourier first described the transform in 1807 while studying heat
// conduction. The fast algorithm was published by Cooley and Tukey in 1965,
// though Gauss had discovered an equivalent method around 1805.
//
// Complexity: O(n log n) for n a power of 2.
func FFT(x []complex128) []complex128 { ... }
```

### Precision documented, not assumed

Every function states:
- Its valid input range
- Its numerical precision
- Its failure modes
- Its golden-file tolerance and why

### No allocations in hot paths

Functions accept output buffers. Functions that must allocate document it. Pistachio calls these at 60 FPS; RubberDuck calls them on every tick.

### Six-tier truth taxonomy

| Tier | Category | Example | Treatment |
|------|----------|---------|-----------|
| 1 | Mathematical Axiom | 1 + 1 = 2 | `const`, no uncertainty |
| 2 | Mathematical Theorem | Pythagorean theorem | Core functions, golden tests |
| 3 | Physical Law | F = ma | Namespaced by domain of validity |
| 4 | Numerical Method | RK4, FFT | Core functions, extensive tests |
| 5a | Exact Constant | c, h | `const` |
| 5b | Measured Constant | G, alpha | `var` with uncertainty |
| 6 | Empirical Relationship | Ohm's law | Documented as approximation |

---

## 8. Key Files in CrossPollinationAnalysis

All source design and review documents live in the CrossPollinationAnalysis repo:

### Design document
- `architecture/UNIVERSAL_TRUTH_FOUNDATION.md` (2,124 lines) -- the complete design

### Review synthesis
- `reviews/reality-review/SYNTHESIS.md` (679 lines) -- distilled consensus from all 9 review agents

### Review files (13 total)

| File | Lines | Agent Role |
|------|------:|------------|
| `reviews/reality-review/philosophy-of-truth.md` | 748 | Epistemologist -- 6-tier taxonomy, inclusion criteria, open-source ethics |
| `reviews/reality-review/scientific-perspective.md` | 600 | Scientist -- what scientists need, adoption bar, build priority |
| `reviews/reality-review/competitive-analysis.md` | 282 | Market analyst -- honest assessment vs gonum/NumPy/Eigen/MathNet |
| `reviews/reality-review/cross-language-strategy.md` | 478 | Systems engineer -- golden-file format, tolerance tables, memory layout |
| `reviews/reality-review/monolith-vs-domains.md` | 590 | Architect -- 4 options evaluated, Option A selected, dependency DAG |
| `reviews/reality-review/chemistry-biology-domains.md` | 129 | Domain specialist -- defer chemistry, partial biology |
| `reviews/reality-review/apps-without-ai.md` | 639 | Product strategist -- 15 pure-reality apps possible without AI |
| `reviews/reality-review/dream-features.md` | 943 | Visionary -- 25 dream features, interactive explorer, equation solver |
| `reviews/reality-review/pistachio-physics-deep-dive.md` | 1,068 | Engine specialist -- 12+ physics gaps, 5-tier integration plan |
| `reviews/reality-review/what-else-is-universal.md` | ~790 | Knowledge surveyor -- 6 new domains approved for reality |
| `reviews/reality-review/foundational-cs-primitives.md` | 680 | CS primitives -- data structures and algorithms audit |
| `reviews/reality-review/foundational-design-patterns.md` | — | Design patterns -- ecosystem patterns audit |
| `reviews/reality-review/ecosystem-bootstrap.md` | 564 | Bootstrap -- 1-import service design |

---

## 9. Implementation Plan Summary

### Phase 0: Golden-file infrastructure (Week 1-2)

Before writing any new math, establish the cross-language test framework.

1. Define the golden-file JSON format with per-function tolerance fields.
2. Write the Go test runner that loads golden files and asserts results.
3. Port the test runner to Python, C++, and C#.
4. Generate golden-file vectors from every existing aicore math function.
5. Run Python/C# equivalents against them. Document every disagreement.

**Deliverable:** Cross-language golden-file framework with existing functions validated.

### Phase 1: Extract and reorganize (Week 3-6)

Move existing math into the reality package structure.

1. Create `github.com/davly/reality` with the sub-package layout.
2. Extract from aicore: echomath -> linalg + stats/information, oraclemath -> stats/bayesian + stats/scoring, calibration -> stats/scoring, parallaxmath -> stats/hypothesis, causalmath -> stats.
3. Update aicore to import from reality.
4. Generate golden files for all extracted functions.

**Deliverable:** reality v0.1.0 with extracted, tested, golden-file-validated math. Zero new functions -- only reorganization.

### Phase 2: Fill the gaps (Week 7-14)

Add core missing functions across all domains.

1. linalg: Full decompositions, sparse matrices, PCA, tensors.
2. calculus: Full ODE suite, optimization (L-BFGS, Nelder-Mead), root finding (Brent).
3. stats: Remaining distributions, MCMC, regression, time series.
4. physics: Classical mechanics, fluid kernels, thermodynamics, E&M.
5. geometry: SDF primitives, quaternions, convex hull, Bezier.
6. signal: FFT, filters, wavelets.
7. constants: CODATA 2022 with uncertainty metadata.
8. graph: Core algorithms, centrality measures.
9. New domains: color, game, decision, queuing, geodesic, sequence.

Each function gets: implementation, golden-file vectors (20-30), precision docs, citation, four language ports.

**Deliverable:** reality v0.5.0 with ~397 functions, ~8,990 golden-file vectors.

### Phase 3: Pistachio and RubberDuck integration (Week 15-18)

Wire reality into the two largest non-Go consumers.

1. Update limitless-cpp with reality headers.
2. Migrate Pistachio internal math (SDF, quaternions, noise) to call reality.
3. Run Pistachio's 1,758 tests for zero regression.
4. Update Limitless.Math NuGet with reality functions.
5. Migrate RubberDuck math (Kalman, Markov, PCA) to call Limitless.Math.
6. Run RubberDuck's 4,399 tests for zero regression.

**Deliverable:** C++ and C# consumers migrated. Cross-language golden files validated end-to-end.

### Phase 4: Open source preparation (Week 19-20)

1. Audit for proprietary references or credentials.
2. Write README with the provenance-and-citation story.
3. Set up CI/CD (GitHub Actions, all 4 languages).
4. Tag reality v1.0.0.
5. Switch repo from private to public.

**Deliverable:** reality v1.0.0, public, MIT licensed.

### Beyond v1.0

- v1.1: Unit-aware computation, uncertainty propagation.
- v2.0: Chemistry domain, advanced biology, control theory, FEM basics.
- Dream features: Interactive explorer, equation solver, formal proofs, REST API.

---

## 10. How This Connects to the Ecosystem

### Dependency inversion: aicore will import reality

Today aicore contains echomath, oraclemath, causalmath, parallaxmath, and calibration -- pure math that has nothing to do with AI. Reality extracts this math. aicore imports reality. The AI layer focuses on what it should: forge, evolution, safety, NexusProvider.

### The chain: every project benefits

```
Consumer App -> limitless-sdk -> aicore -> reality
```

One bug fix in reality/stats propagates to every project that uses statistics. One precision improvement reaches all four languages through golden files.

### Specific ecosystem connections

**Pistachio gets real physics.** The Pistachio deep-dive identified 12+ physics gaps: N-body gravity, aerodynamic drag, springs, SPH fluids, soft bodies, smoke/fire, heat transfer, sound propagation. Reality provides the mathematical kernels; Pistachio provides the GPU dispatch and rendering.

**RubberDuck gets real stochastic calculus.** Kalman filters, Markov chains, HMM, ARIMA, PCA -- all validated against golden files with known precision bounds. Kelly criterion from reality/decision for position sizing.

**Oracle gets real Bayesian inference.** Conjugate updates, MCMC, calibration scoring -- extracted from oraclemath into reality/stats with full golden-file coverage.

**Echo gets real topology.** Persistent homology, Betti numbers, Wasserstein distance -- the TDA pipeline backed by validated linear algebra.

**Sentinel gets queuing theory.** Alert thresholds informed by M/M/1 models. Utilization above 0.8 means nonlinear wait time growth -- now backed by real mathematics.

**Every consumer app gets correct color.** WCAG contrast ratios, perceptual color distance, proper gamma handling -- from reality/color.

### 15 pure-reality apps possible without AI

The apps-without-AI review identified consumer applications buildable entirely on reality with no AI, no network calls, no per-request cost: physics simulation (Forge), scientific calculator (Abacus), signal processing workstation (Wavelength), statistics dashboard (Census), astronomical simulation (Orrery), financial modeling (Ledger Prime), engineering calculator (Girder), music theory tool (Temperament), terrain generator (Terraform), graph analyzer (Lattice), and more.

### The consistency problem is solved

Today, a Brier score computed by Oracle (Go) and one computed by Horizon (Python) might disagree by 1e-14. After reality, the golden-file suite answers definitively: same formula, same inputs, same output, four languages.

### The single most important design decision

**The golden-file test infrastructure.** Not the functions. Not the architecture. Not the license. The golden files.

Everything else follows:
- Cross-language consistency exists because the golden files enforce it.
- Correctness is verifiable because the golden files define it.
- Provenance is meaningful because the golden files pin expected output to cited formulas.
- Regression is caught because the golden files never change unless the math does.
- Trust is earned because the golden files are public, inspectable, and reproducible.

If reality ships 397 functions with mediocre golden files, it is just another math library. If it ships 20 functions with perfect golden files -- 30 vectors each, all IEEE 754 edge cases covered, all four languages validated, per-function tolerances documented -- it is the foundation that everything else builds on.

The golden files are the proof. Start there. The functions will follow.

---

## Appendix: Decision Log

| Decision | Selected | Rationale |
|----------|----------|-----------|
| Repository structure | Monolith (Option A) | Simplest deps, matches aicore precedent, lowest maintenance |
| Canonical language | Go | Largest test suite, existing golden-file pattern, source of truth |
| License | MIT | Universal truths should be universally accessible |
| Chemistry in v1.0 | Defer | No new primitives; pure applied physics |
| Biology in v1.0 | Partial | Pop dynamics and epidemiology are pure ODEs |
| Units in v1.0 | Defer to v1.1 | Avoid delaying core math for cross-language type design |
| Physics scope | Equations only | Reality = functions, Pistachio = engine |
| Open source timing | v1.0 | Credibility signal, correctness benefit, ethical alignment |
| Starting point | Extract existing | 1,044 aicore tests + 358 limitless-py tests already battle-tested |
| Golden-file tolerance | Per-function | Transcendentals need 1e-11; accumulating ops need 1e-9 |
| New domains (color, game, decision, queuing, geodesic, sequence) | Include in reality | Mathematics wearing domain-specific names; multi-consumer; passes inclusion tests |

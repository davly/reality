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

## 3. The Packages (24)

Reality currently contains 24 sub-packages. The original design called for 16; the roster grew as the physics scope was decomposed into focused packages, applied-math areas were identified, and Session 23/24 added `sequence` (string distance, alignment, n-grams) and a `conduit` emit shim. The 22-package table below lists the v1.0 core domains; see Section 11 for `sequence` and `conduit`.

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

---

## 11. Session 25 State (2026-04-08)

This section is the living "current state" layer on top of the v1.0 design above. The earlier sections describe the architecture as-decided; this section describes the repository as-built on 2026-04-08.

### Version and build

| Field | Value |
|---|---|
| Version | v0.10.0 (README, CLAUDE.md) |
| Go module | `github.com/davly/reality` |
| Go version required | 1.24+ |
| License | MIT (documented in README; a LICENSE file is not yet committed) |
| External dependencies | **Zero** — only Go stdlib (`math`, `math/cmplx`, `math/rand`, `sort`, `errors`, `strings`, `bytes`, `container/heap`, `encoding/json`, `os`, `runtime`, `path/filepath`, `testing`, `context`, `net/http`, `sync/atomic`, `time`, `strconv`) |
| Branch | `master` (not yet renamed to `main`) |
| GitHub remote | `https://github.com/davly/reality.git` (the ONLY public repo in the Limitless ecosystem) |

### Built packages (44 as of 2026-05-01; was 24 at Session 25 snapshot)

The v1.0 design table lists 22 domain packages. Many more have landed since the Session 25 snapshot — the headline additions:

| Added | Package | Purpose | Source files |
|---|---|---|---|
| Session 23 (commit `07503a3`) | `reality/sequence` | Edit distances, sequence alignment, n-grams. Wagner-Fischer Levenshtein, Needleman-Wunsch, Smith-Waterman, Hamming, Jaro-Winkler, n-gram extraction. | `distance.go`, `alignment.go`, `ngram.go` |
| Session 24 (commit `7709936`, Wave 6.A5) | `reality/conduit` | Fire-and-forget HTTP shim publishing ForgeEcosystemEvents to the Conduit bus. Non-blocking, 100ms timeout, 1-in-N sampling for hot-path math primitives. This is the ONE package in reality that is NOT pure math. | `emit.go` (no tests — trivial shim) |
| 2026-05-01 (commit `edd1428`, audio cohort intake; refined `260ea33`) | `reality/audio` (`audio/melscale`, `audio/mfcc`, `audio/fingerprint`, `audio/degradation` content) | Spectral substrate for the audio-trio flagships (Pigeonhole / Howler / Dipstick): mel filterbank (Slaney 1998), MFCC (DCT-II orthonormal, HTK convention), Welford-based fingerprint (1962, with Chan-Golub-LeVeque 1979 merge), DegradationTracker (Shewhart/EWMA control-chart on a frozen baseline). | `melscale.go`, `mfcc.go`, `fingerprint.go`, `degradation.go` |
| 2026-05-01 (substrate-promotion from `flagships/dipstick/reference/forge/vibration.go`) | `reality/audio/vibration` | Mechanical vibration primitives: `FundamentalHz` (peak-bin fundamental from windowed FFT) + `HarmonicEnergyRatio` (fraction of total band-power in harmonic bands). First instantiated consumer = Dipstick; Fleetworks Torque expected 2nd; KMM mobile port the 3rd (SHARED-ENGINE-DUAL-BRAND R-pattern, 1/3). | `doc.go`, `fundamental.go`, `harmonic.go`, `vibration_test.go` |
| 2026-05-01 (cohort review §14.P closure) | `reality/audio/separation` | Multi-source audio signal-separation primitives — the cocktail-party-problem solution set: spectral subtraction (Boll 1979), Wiener filter (Wiener 1949 / Ephraim-Malah 1984), FastICA (Hyvärinen 1999), NMF (Lee & Seung 1999/2001 multiplicative-update), Energy-VAD + ZCR (Rabiner & Schafer 1975). Used by Pigeonhole's "3 birds singing at once" feature, Howler multi-pet isolation, Dipstick multi-component machine separation. | `doc.go`, `spectral_subtraction.go`, `wiener.go`, `ica.go`, `nmf.go`, `vad.go`, `separation_test.go` |
| 2026-05-01 (cohort review §14.P closure) | `reality/audio/spectrogram` | STFT computation + visualisation primitives: `Compute` (overlap-add forward STFT), `Inverse` (Griffin-Lim OLA reconstruction), `Magnitude` / `LogMagnitude` / `PowerSpectrum`, `MelSpectrogram` / `LogMelSpectrogram` (composes audio.MelFilterbank), `ToHeatmap` (PNG-encoded heatmap output via std-lib `image/png`), production colourmaps Plasma / Magma / Viridis / Inferno (matplotlib-compatible CC0 LUTs). Used by Pigeonhole / Howler / Dipstick spectrogram-as-art rendering. | `doc.go`, `stft.go`, `magnitude.go`, `mel_spectrogram.go`, `colourmap.go`, `visualise.go`, `spectrogram_test.go` |
| 2026-05-01 (audio-cohort round-out, this commit) | `reality/audio/onset` | MIR onset-detection primitives: `EnergyOnset` (RMS-rise; Schloss 1985 / Klapuri 1999), `SpectralFluxOnset` (Bello & Sandler 2003 / Dixon 2006), `ComplexDomainOnset` (Bello et al. 2004 — magnitude + phase residual), `SuperFlux` (Böck & Widmer 2013 — vibrato-suppressing max-filter SF), `PickPeaks` / `PickPeaksAdaptive` (median-filter peak picker). Localises percussive events (bird-call onsets, mechanical service-events, drum hits). | `doc.go`, `energy.go`, `spectral_flux.go`, `complex_domain.go`, `superflux.go`, `peak_picking.go`, `onset_test.go` |
| 2026-05-01 (audio-cohort round-out, this commit) | `reality/audio/segmentation` | Audio event-segmentation primitives — extracts individual events (one bird call, one valve actuation) from longer recordings: `SegmentByEnergy` (energy-VAD; Rabiner & Sambur 1975), `SegmentByOnsetOffset` (spectral-flux onset + energy-decay offset; Bello et al. 2005), `SegmentWithMinSilence` (silence-duration-thresholded splitting), `MergeCloseSegments` (post-process coalescer), `FilterByMinDuration` (post-process noise dropper). The Pigeonhole "one call at a time" workflow lives here. | `doc.go`, `vad_based.go`, `onset_offset.go`, `min_silence.go`, `merge_close.go`, `min_duration.go`, `segmentation_test.go` |
| 2026-05-01 (audio-cohort round-out, this commit) | `reality/audio/pitch` | Pitch / fundamental-frequency estimators — more robust than `audio/vibration`'s FFT peak-picker for vocal / bird / pet signals: `AutocorrelationPitch` (classical ACF; Roads 1996 §10), `Yin` (de Cheveigné & Kawahara 2002 — de-facto modern monophonic pitch detector with parabolic-interpolated sub-sample precision and aperiodicity), `McLeodPitchMethod` (McLeod & Wyvill 2005 — normalised-SDF with clarity), `SubharmonicSummation` (Hermes 1988 — frequency-domain harmonic-summation; detects missing fundamentals). | `doc.go`, `autocorrelation.go`, `yin.go`, `mpm.go`, `subharmonic_summation.go`, `pitch_test.go` |

Note that the design document (`architecture/UNIVERSAL_TRUTH_FOUNDATION.md` in LimitlessGodfather) does not mention `sequence`, `conduit`, `audio`, `audio/vibration`, `audio/separation`, `audio/spectrogram`, `audio/onset`, `audio/segmentation`, `audio/pitch`, or several other post-Session-25 packages (`autodiff`, `changepoint`, `forge`, `info`, `info/{lz,mdl}`, `infogeo`, `pkg/...`, `prob/conformal`, `prob/copula`, `timeseries`, `topology`); they post-date it. `conduit` is a minor philosophical exception to "zero dependencies, pure math" — it makes a `net/http` call and accepts that math primitives are now *observable* by the Conduit bus (though only via sampled emit, default 1-in-10,000). Reality itself does not call its own `conduit` package in any hot path in the current tree; it is present for callers that want to emit lifecycle events. `audio/vibration` is the substrate-extraction outcome of Dipstick's reference forge — see `flagships/dipstick/docs/INSIGHTS.md` §3 for the SHARED-ENGINE-DUAL-BRAND R-pattern this package anchors. `audio/separation` and `audio/spectrogram` close the last two audio substrate gaps noted in the cohort review §14.P (multi-source separation; STFT visualisation pipeline).

### Test counts

| Metric | v1.0 plan | README claim | Actual (Session 25) |
|---|---|---|---|
| Packages | 22 core | 22 | **24** (22 core + `sequence` + `conduit`) |
| Top-level test functions (`--- PASS`) | — | 1,965 | **1,585** |
| Total test runs (`=== RUN` including subtests) | — | — | **2,409** |
| Non-test Go LOC | — | — | ~15,400 |
| Test Go LOC | — | — | ~19,300 |
| Public non-test functions | ~397 | — | **426** |
| Golden-file JSON fixtures (`testdata/**.json`) | ~8,990 vectors | — | **73 fixture files** (each holding many vectors) |

The 1,965 figure in `README.md` and `CLAUDE.md` counts test *invocations* from an older toolchain run; the current `go test -v ./...` reports 1,585 top-level `--- PASS` lines and 2,409 `=== RUN` entries including subtests. All packages build clean, all tests pass, no regressions.

### Per-package test count (top-level PASSes, Session 25)

| Package | Tests | Package | Tests |
|---|---:|---|---:|
| linalg | 154 | physics | 79 |
| prob | 151 | control | 78 |
| graph | 134 | fluids | 70 |
| sequence | 122 | signal | 69 |
| geometry | 91 | gametheory | 66 |
| physics | 79 | optim | 63 |
| control | 78 | queue | 63 |
| combinatorics | 58 | acoustics | 62 |
| compression | 57 | chaos | 51 |
| crypto | 50 | color | 49 |
| calculus | 45 | em | 31 |
| orbital | 27 | testutil | 10 |
| constants | 5 (golden-file driven) | conduit | 0 (shim, no tests) |

(`constants` has only 5 tests because its work is the golden-file table; `testutil` has 10 because it is the infrastructure those golden files use.)

### Golden-file fixture inventory (73 files)

Golden files are not uniformly distributed. Packages with domain packages that host vectors: `acoustics` (4), `color` (2), `combinatorics` (1), `compression` (1), `em` (10), `fluids` (5), `geometry` (2), `graph` (7), `linalg` (5), `orbital` (8), `physics` (3), `prob` (7), `signal` (1), plus `testutil` (2 samples). Additional centralised vectors under top-level `testdata/` exist for `calculus` (4), `chaos` (1), `constants` (1), `control` (1), `crypto` (2), `gametheory` (2), `optim` (1), `queue` (2), `sequence` (1). Coverage is clearly uneven — `linalg`, `prob`, `em`, `orbital`, and `graph` are well represented; `signal` has one fixture for `fft`, `combinatorics` has one for `binomial_coeff`. This is a known expansion area.

### Recent trajectory (git log, most recent first)

| Commit | Meaning |
|---|---|
| `a3cc0f7` | session25(audit): auto-generated `ARCHITECTURE.md` baseline (Session 25 replaces this) |
| `7709936` | wave6(reality): conduit-emit shim — Session 24 Wave 6.A5 addition |
| `5b63907` | Session 9: edge-case test coverage for under-tested packages |
| `809138f`, `c41d19e` | Docs updates for v0.10.0 (22 packages, 1,413 → 1,965 tests) |
| `26d4814` | Stirling numbers added to combinatorics |
| `27113e2` | MatSub, Trace, FrobeniusNorm added to linalg |
| `edf961b` | MarkovChain, GammaPDF/CDF added to prob with golden vectors |
| `82afb84` | PageRank, BellmanFord, KruskalMST added to graph with golden vectors |
| `07503a3` | `sequence` package added (14 functions, 55 tests) — Session 23 |
| `c389b6f` | `em` package added (10 functions, 32 tests) |
| `e94347e` | `orbital` package added (8 functions, 27 tests) |
| `b27204c` | `acoustics` package added (9 functions, 37 tests) |
| `7903924` | `fluids` package added (10 functions, 38 tests) |
| `73c79b4` | Genetic algorithm + simplex method optim extensions |
| `46d0f38` | LinearRegression, FisherExactTest, MannWhitneyU, BenjaminiHochberg added to prob |
| `2f5c7b7` | `calculus` package added (6 functions, 45 tests) |
| `f3bc1cb` | Graph algorithms (Dijkstra, A*, betweenness, Louvain, max flow, topsort) |
| `176c316` | Phase 6a: `chaos` (ODE solvers, Lorenz, SIR, Lotka-Volterra, Lyapunov) |
| `bdb133d` | Phase 5b: `gametheory` (Nash, Gale-Shapley, bandits, voting, Kelly) |
| `921dbe2` | Phase 5c: `queue` (M/M/1, M/M/c, Erlang B/C, Jackson networks) |
| `2455b19` | Phase 5a: `control` (PID, filters, transfer functions) |

The build has been additive since Phase 2 — every commit lands a new domain or extends an existing one with golden-file coverage. No regressions have been reverted. Rename from Phase-numbered commits to feature-scoped commits happens at commit `2f5c7b7`.

### Downstream consumers

Reality is the only Tier 0 foundation in the ecosystem. Direct importers today (via `github.com/davly/reality/*`):

- **aicore** (`github.com/davly/aicore/*`) — pulls `linalg`, `prob`, `crypto` for its echomath/oraclemath/causalmath modules. This is the dependency inversion described in Section 10; extraction has been partial — some aicore subpackages still hold forked copies.
- **Pistachio (C++)** — per design, should consume `limitless-cpp` headers that mirror reality functions (geometry, physics, signal, constants). Port status is "planned, not yet shipped" as of the last L3 audit.
- **RubberDuck (C#)** — per design, should consume `Limitless.AI.Reality` mirror. Port status: not shipped.
- **Every Go flagship that needs math** — `echo`, `oracle`, `parallax`, `horizon`, `phantom`, `sentinel`, `nexus`, `delve`, `synthesis`, `causal`, `recall`, `grounded`, `pulse`, `paradox` — any of these that want FFT, FNV-1a (`crypto/hash.go`), Jeffreys prior math, convergence scoring, or probability primitives should import from reality directly. Current state is mixed: `crypto.FNV1a64` is canonical across the ecosystem (Wave 6.A1), but Jeffreys/Dominance helpers still live in individual flagships rather than `reality/prob`.

### What Session 25 should know

Open items that did not exist when the v1.0 design was written:

1. **Canonical FNV-1a lives here.** `reality/crypto/hash.go` holds `FNV1a32` and `FNV1a64`. The Session 22 FNV-1a cross-language fix (P0) canonicalised these; `architecture/fnv1a_canonical_vectors.json` in LimitlessGodfather holds the golden vectors. Any other FNV-1a implementation in the ecosystem should delegate here.
2. **`sequence` package** is newer than the design doc and should probably get a first-class entry in Section 3 on the next CONTEXT.md revision. It provides 14 functions, 122 tests (!) and is heavily used by flagships doing approximate matching.
3. **`conduit` package** is the philosophical exception. Reality is no longer *strictly* zero-dependency on `net/http` if you count the emit shim. Treatment: the shim is isolated in its own package so consumers that care about purity can simply not import it. No hot path in reality itself calls it.
4. **Jeffreys prior helpers are missing.** The ecosystem convergence standard (Session 23) mandates Jeffreys (0.5, 0.5) with quality-weighted dominance, but the reference implementation lives in individual flagships and in `infrastructure/delve` rather than `reality/prob`. A `reality/prob/jeffreys.go` file with `BetaJeffreysDominance`, `QualityWeightedDominance`, and `ThreeWayVerdict` primitives is the obvious next addition — it would let every flagship delete its local copy.
5. **Non-Go ports have not shipped.** Python `limitless-py/reality.py`, C++ `limitless-cpp/reality.hpp`, and C# `Limitless.AI.Reality` are all still in the "planned" column. Session 25's audit should record this as a P1 gap.
6. **Golden-file coverage is uneven.** 73 fixture files is fewer than one per non-test `*.go` source file. Sparse packages: `signal` (1 fixture for FFT, none for filters or windows), `combinatorics` (1 fixture for binomial, none for Stirling/Bell/partitions), `chaos` (1 fixture for Lorenz). Expansion would strengthen the cross-language guarantee.
7. **Branch is still `master`.** Per the ecosystem memory, ~70 repos need `master → main`. Reality is one of them.
8. **LICENSE file is not committed.** The README and CONTEXT.md both declare MIT. Adding a `LICENSE` file is a zero-risk P2 action.

### What is unambiguously working

- All 24 packages build under Go 1.24 with `go build ./...`.
- All tests pass under `go test ./...` on 2026-04-08.
- Zero external dependencies — `go.mod` has no `require` clauses beyond the module declaration.
- The module builds on Windows, Linux, and macOS without any build tags or platform-specific code (no `//go:build` directives in non-test files).
- FNV-1a, Normal PDF/CDF/Quantile (Acklam), FFT (Cooley-Tukey radix-2, in-place, zero-alloc), Dijkstra/A*, Kepler orbital, CIEDE2000, and ~420 other functions are all documented, tested, and cited.

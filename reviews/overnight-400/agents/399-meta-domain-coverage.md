# 399 — meta-domain-coverage

## Headline
Reality covers ~18 of MSC2020's ~60 top-level areas at "average or stronger"; the truly mission-critical gaps are special functions (MSC 33), PDE (MSC 35), measure/integration (MSC 28), and AI/ML attention/RL primitives — every other "absent" MSC area is genuinely out-of-scope for a 4-language golden-file library.

## Inventory recap (current as of 2026-05-09)

CLAUDE.md still says "22 packages, 1,965 tests" — actual repo has **41 top-level dirs** with ~37 production packages (slot 382 confirmed). Material additions since the doc-stamp:
`audio/{root,beat,cqt,onset,pitch,segmentation,separation,spectrogram,tempo,vibration}`, `autodiff`, `changepoint`, `conduit`, `forge/session40`, `info/{lz,mdl}`, `infogeo`, `optim/{proximal,transport}`, `pkg/canonical`, `prob/{copula,conformal}`, `sequence`, `timeseries/{dcc,garch}`, `topology/persistent`, `zkmark`.

Effectively 35 production math packages + 6 plumbing/test (`testutil`, `conduit`, `forge`, `pkg`, `docs`, `reviews`).

## Coverage matrix (MSC 2020 top-level vs reality)

Status legend: **STRONG** (deeper than gonum/peers, multiple sub-packages), **AVG** (functional, breadth-shaped), **WEAK** (token presence), **ABSENT** (no code), **OOS** (out-of-scope by mission).

| MSC area | Status | reality packages | Notes |
|---|---|---|---|
| 03 Logic / set theory | OOS | — | No formal proof / set-theory consumer; gnark-style symbolic logic lives elsewhere |
| 05 Combinatorics | AVG | `combinatorics/` | Counting, perms, Catalan, Stirling, Bell. Slot 38 noted Heap iterator gap |
| 06 Order, lattices | ABSENT | — | No lattice/poset primitives; minor gap (could matter for `infogeo`/MDL but no consumer pull) |
| 08 General algebraic systems | OOS | — | |
| 11 Number theory | AVG | `crypto/` | Primality (MR, Mersenne), modular arith, totient. Slot 290 (Galois) flagged extension-field absence |
| 12-13 Field/commutative algebra | ABSENT | — | Galois theory absent (slot 290); blocks finite-field crypto + Reed-Solomon |
| 14 Algebraic geometry | OOS | — | |
| 15 Linear & multilinear algebra | AVG | `linalg/` | LU/QR/Cholesky/PCA/sparse. Behind gonum on depth (slot 357, slot 374 BLAS-modern). Tensor algebra (>2 indices) absent |
| 16-18 Category theory | OOS | — | |
| 19-20 K-theory, group theory | OOS | — | (Quaternion is in `geometry`, not abstract group theory) |
| 22 Topological groups, Lie | WEAK | `geometry/` quaternion only | SO(3) / SE(3) / Lie-algebra `exp`/`log` absent — robotics/IMU consumers want this (slot 381 also flagged Rotation3 type) |
| 26 Real functions | AVG | `calculus/` | Numerical diff, Simpson, RK4, root-find |
| 28 Measure & integration | WEAK | `calculus/` quadrature | No abstract measure, no Lebesgue / Stieltjes; matters for rigorous prob theory but no immediate consumer |
| 30-32 Complex analysis | WEAK | `signal/` Hilbert, FFT | No analytic-function tooling, no conformal maps. Largely OOS |
| 33 Special functions | **WEAK** | `prob/mathutil.go` (Γ, B, erf), `prob/copula/studentt.go` (Hill 396) | **Bessel, spherical harmonics, hypergeometric, elliptic, Mathieu — all absent.** Slot 300 (Bessel-spherical) is open. **Critical gap** — see Top-5 #1 |
| 34 ODE | STRONG | `chaos/`, `physics/thermo.go` | RK4Step + SolveODE pair, Lorenz, Van der Pol, Lyapunov. Adaptive Dormand-Prince absent (slot 28 noted) |
| 35 PDE | **ABSENT** | `physics/thermo.go` 1-D heat eq only | **No FEM, FDM, FVM, spectral methods.** Slot 250 (mortar) identified the gap. Blocks fluid-sim, EM full-wave, real acoustic-room sim |
| 37 Dynamical systems | STRONG | `chaos/` | Strong against Go peers; only gosl approaches |
| 39 Difference & functional eqs | WEAK | `timeseries/{garch,dcc}` | Recurrence machinery exists; no general difference-eq solver |
| 40-46 Series, harmonic, real/functional analysis | WEAK | `signal/` only | FFT/IFFT/window/Hilbert. Wavelets absent (slot 134 signal-api). Function-space norms absent |
| 47 Operator theory | OOS | — | |
| 49 Calculus of variations / optimal control | AVG | `optim/`, `control/` | No proper optimal-control (LQR, MPC) — slot 230 controls flagged. Calculus of variations absent |
| 51-53 Geometry (Euclidean, projective, differential) | AVG | `geometry/` | Quaternions, SDF, curves, hull, projective. Differential geometry (curvature, parallel transport) absent |
| 54 General topology | WEAK | `topology/persistent/` | TDA only — no point-set topology utilities (compactness, etc.); fine — TDA is the consumer-facing piece |
| 55 Algebraic topology | AVG | `topology/persistent/` | Vietoris-Rips, barcode, bottleneck. Reeb/Mapper open (slots 286-287) |
| 57-58 Manifolds | WEAK | `geometry/` quaternion + autodiff | No Riemannian-optimization manifold types (Stiefel, Grassmann); slot 381 noted defer-justifiable |
| 60 Probability | STRONG | `prob/`, `prob/copula/`, `prob/conformal/` | Distributions, Bayesian, copulas, conformal — broader than gonum (slot 357) |
| 62 Statistics | STRONG | `prob/`, `infogeo/`, `changepoint/` | Hypothesis, regression, nonparametric, BOCPD |
| 65 Numerical analysis | AVG | `linalg/`, `calculus/`, `optim/` | All-rounder; no IEEE-754 corner-case discipline (slot 377 num-stab) |
| 68 CS / algorithms | STRONG | `graph/`, `compression/`, `sequence/`, `info/{lz,mdl}` | Dijkstra/A*/topo + edit-distance + LZ + MDL. Suffix automaton, Rabin-Karp absent (slot 398) |
| 70 Mechanics of particles | AVG | `physics/`, `orbital/` | Kepler, vis-viva, classical mechanics |
| 74 Mechanics of solids | WEAK | `physics/materials.go` | Stress/strain only; FEM absent — see Top-5 #2 |
| 76 Fluid mechanics | AVG | `fluids/` | Reynolds, Bernoulli, Darcy-Weisbach, drag. **No CFD** — algebraic primitives only |
| 78 Optics, EM theory | AVG | `em/`, `physics/optics.go` | Coulomb, Ohm, RC/LC. **No FDTD / Maxwell-solver** |
| 80 Classical thermodynamics | AVG | `physics/thermo.go` | Stefan-Boltzmann, Newton cooling, ideal gas |
| 81 Quantum theory | ABSENT | — | OOS for 2026 — no consumer demand inside aicore/Pistachio |
| 82 Statistical mechanics | ABSENT | — | OOS |
| 83 Relativity | ABSENT | — | OOS — no Vocala consumer asks |
| 85 Astronomy / astrophysics | AVG | `orbital/` | Kepler, Hohmann, escape velocity, Hill sphere |
| 86 Geophysics | OOS | — | |
| 90 Operations research / game theory | STRONG | `gametheory/`, `queue/`, `optim/transport/` | Nash, Shapley, Kelly, bandits, M/M/c, Erlang, Sinkhorn — broader than any single Go peer |
| 91 Behavioral / social sci math | AVG | `gametheory/bandit.go`, `prob/markov.go` | Markov chains, multi-armed bandits |
| 92 Biology | ABSENT | — | OOS |
| 93 Systems & control | AVG | `control/` | PID, transfer, Bode, stability. No state-space, LQR, MPC — slot 230 |
| 94 Information & comms | STRONG | `info/{lz,mdl}/`, `infogeo/`, `compression/`, `prob/` (entropy) | Shannon entropy, KL, f-div, Fisher metric, Bregman, MDL/NML, LZ76 — uniquely deep across Go ecosystem |
| **Audio DSP** (not MSC) | STRONG | `audio/{onset,beat,tempo,pitch,cqt,...}`, `signal/` | Pistachio-driven; 3-detector cross-validation, MFCC, CQT, MPM/YIN, vibration |
| **Color science** (not MSC) | STRONG | `color/` | 8 spaces, CIEDE2000, Bradford — unique in Go |
| **Crypto/ZK** (applied) | AVG | `crypto/`, `zkmark/` | Hash, PRNGs, primality + a ZK-mark primitive |
| **Autodiff** (ML primitive) | AVG | `autodiff/` | Reverse-mode tape, copula gradient pin (commit 365368a). Forward-mode (dual numbers) absent — see Top-5 #3 |
| **Attention / transformer kernels** | **ABSENT** | — | Critical for aicore — see Top-5 #4 |
| **RL primitives (policy grad, TD, PPO core)** | WEAK | `gametheory/bandit.go` only | Bandits exist; tabular Q / TD / policy-grad absent — see Top-5 #5 |

## Strong domains (count)

STRONG: probability/copula/conformal (60-62), graph/algorithms (68), info-theory/info-geometry (94), audio-DSP, color, dynamical systems (37), game-theory/OR (90).

7 STRONG cells. Each is a defensible moat vs gonum (slot 357) and the `gosl`/`gorgonia`/`mgl` archipelago.

## Top 5 strategic gaps (relative to mission, blocked-consumer framing)

### 1. Special functions (MSC 33) — Bessel, spherical harmonics, elliptic
**Blocks:** `acoustics` cylindrical-room modes, `em` waveguide cutoffs, `orbital` perturbation theory, **Pistachio** spherical-harmonic ambisonics. Slot 300 (Bessel-spherical) explicitly opens this. Without J_n, Y_n, I_n, K_n, P_l^m the library cannot speak honestly about wave physics. Reference impls: SciPy (Cephes), Boost.Math, GSL. Golden-file portable since they are pure float64.

### 2. PDE solvers (MSC 35) — at minimum FDTD-1D/2D + FEM-1D
**Blocks:** any "real" `acoustics` room sim (vs Sabine algebraic estimate), any `em` full-wave, any `fluids` CFD beyond Bernoulli, `physics` heat conduction beyond `HeatEquation1DStep`. Slot 250 (`mortar`/methods) flagged this. Even a 200-line 2-D FDTD wave-equation core would unlock 3 packages simultaneously. Closest peer: `gosl` via FFI. Reality has a clean shot at first-class pure-Go FDTD with golden-file determinism — nobody else in Go does.

### 3. Forward-mode autodiff (dual numbers) — complements existing tape
**Blocks:** `aicore`'s small-input/large-output gradient cases (e.g. ESS sensitivity analysis on a single hyperparameter), Pistachio's per-sample 60 FPS gradient where reverse-mode tape allocation is too costly. Slot 398 (streaming-vs-batch) noted "forward-mode would be the truly streaming variant." `autodiff/` already has reverse-mode (`tape.go`); ~300 LOC dual-number package + golden file would close the gap.

### 4. Attention / transformer / softmax-with-mask kernels
**Blocks:** `aicore` transformer inference paths that today route around reality. CLAUDE.md says "be a foundational math library for AI/audio/graphics/control/sim apps" — currently the AI side is the weakest. Even a `attention/` package with: scaled-dot-product, softmax with numerical-stability subtraction, masked-softmax, RMSNorm, RoPE, GeLU/SiLU/Swish, would let aicore drop a hand-rolled `safe_softmax` it likely maintains. Slot 220-F (modern optimizers — Adam/AdamW/Lion) is the natural sibling; SGD/Adam are also absent.

### 5. RL primitives — TD(0), Q-learning, policy gradient, GAE
**Blocks:** any aicore/Vocala agent that wants tabular or function-approx RL without pulling in Python. Bandits (`gametheory/bandit.go`) cover stateless exploration; full RL needs a value/policy update primitive set. ~500 LOC. The MDP plumbing (states, actions, transition, reward) shares structure with `prob/markov.go`, so partial code reuse is plausible.

## Top 3 over-served areas (lots of code, no observed consumer)

### 1. `orbital/` (MSC 85)
Kepler, vis-viva, Hohmann, escape velocity, Hill sphere — full breadth. **No internal consumer:** Pistachio is audio, aicore is LLM/agent, Vocala is voice-platform. Astronomy isn't on the consumer roadmap. Code is correct and tested but earns its keep only as a "we are a real physics library" marketing surface. Suggest: keep but freeze; do not invest in perturbation theory or n-body unless a real consumer appears.

### 2. `em/` and `fluids/` (MSC 76 / 78)
Both are algebraic-primitive packages: Coulomb, Ohm, RC, Reynolds, Bernoulli, Darcy-Weisbach. They cover the "physics 101 textbook formulas" surface without any solver depth. Same consumer-vacuum diagnosis as `orbital/`. The right move is **not** to delete them but to recognize the asymmetry: `acoustics/` has a Pistachio consumer (audio chain), `em/`+`fluids/` do not. Investment priority: low until/unless a sim/AR consumer materializes.

### 3. `gametheory/` matchings + Shapley
Nash, Shapley, replicator dynamics, stable matching — broader than any Go peer. The bandit.go subset *is* consumed (any reinforcement-style explorer); the cooperative-game-theory subset (Shapley, core, Banzhaf) has no observed reality consumer. Bandits earn their keep; Shapley is academic completeness.

## Recommendation

- **Invest top-line:** special functions (slot 300 ready), PDE/FDTD-2D (slot 250 ready), attention kernels + Adam optimizer (slot 220-F ready), forward-mode autodiff. Each has an explicit consumer (Pistachio, aicore) and an MSC-area justification.
- **Hold steady:** `prob/*`, `info/*`, `infogeo/*`, `audio/*`, `graph/*`, `color/*`, `linalg/*`, `signal/*` — the seven STRONG cells. Apply slot 382 backlog (golden-vector counts, fuzz, IEEE 754 specials) here first; depth before breadth.
- **Freeze, don't extend:** `orbital/`, the algebraic surface of `em/` and `fluids/`, `gametheory/` cooperative-game functions. Maintain golden files; do not solicit feature requests.
- **Update CLAUDE.md** to reflect 35-37 packages, not 22 (slot 382 already noted). Add a "consumer-pull index" table so future agents can re-prioritize without re-deriving.
- **Re-classify before adding new MSC areas.** MSC 81 (QM), 82 (stat-mech), 83 (relativity), 92 (biology) are correctly OOS today. If a new consumer (e.g. a quantum-sim demo) arrives, formally promote one OOS area at a time.

## Sources

- `C:/limitless/foundation/reality/CLAUDE.md` — mission, package list, design rules
- Slot 357 `research-libs-go.md` — gonum coverage matrix, USP analysis
- Slot 381 `meta-types-system.md` — type-policy inventory, dimension-confusion ranking
- Slot 382 `meta-test-coverage.md` — actual 41-package count, growth-package golden gaps
- Slot 389 `meta-doc-cohesion.md` — public-function counts (~676), citation coverage
- Slot 398 `meta-streaming-vs-batch.md` — per-package streaming inventory and gap calls
- Slot 250 `new-mortar.md` — PDE/methods opening
- Slot 290 `new-galois-theory.md` — finite-field gap
- Slot 300 `new-bessel-spherical.md` — special-function opening
- Slot 230 (controls modern) — LQR/MPC absence
- Slot 220-F (optim modern) — Adam/AdamW/Lion absence
- Slots 286-287 (topology Reeb/Mapper) — TDA depth gaps
- MSC 2020 classification: https://msc2020.org/, https://zbmath.org/classification/

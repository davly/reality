# 088 | info-sota — SOTA information-theory library comparison

**Scope.** Survey of 8 reference information-theory libraries — JIDT,
IDTxl, dit, NPEET (+ NPEET_LNC), pyitlib, infomeasure, PyPhi, POT,
mpmath — for the express purpose of identifying (a) the *headline
algorithm* each ships, (b) the *engineering trick* worth borrowing,
(c) the *zero-dependency portability* path into `reality/` (Go-only,
no math beyond stdlib + `math/big`).

**Why this scope, not 086/087's.** 086 audited the numerics of what
`reality` already has. 087 enumerated which canonical primitives are
*missing*. This report goes one level out: who is the SOTA reference
implementation for each missing primitive, what concrete code-shape
trick they use that beats the textbook formula, and what survives the
"zero-dep, pure Go, no SciPy/NumPy" portability filter that is
`reality`'s defining constraint.

**Filter applied.** Every "borrow" recommendation below is checked
against three constraints:
1. Implementable in pure Go with stdlib + `math/big` only (no FFTW,
   no LAPACK, no GPU, no Java JNI, no Python ctypes).
2. ≤ ~500 LOC for the standalone primitive (Tier 3 caps).
3. Reproducible against a golden-file (deterministic output, no
   hidden randomness, or seedable + documented).

A library that fails (1) is not unusable as a *reference* — it is
just unusable as a *port*; we then borrow its algorithm, not its
code. JIDT, IDTxl, PyPhi all fail (1) directly but supply
gold-standard test vectors.

---

## Library-by-library breakdown

### 1. JIDT — Java Information Dynamics Toolkit (Lizier 2014)

- **Headline algorithm.** Multivariate Kraskov-Stögbauer-Grassberger
  (KSG-1 and KSG-2) mutual information, conditional MI, transfer
  entropy, conditional TE, active information storage, predictive
  information, all sharing the same kNN backend.
- **Engineering trick.** Single multivariate k-d tree built once over
  the joint embedding `(X_past, Y_past, Y_future)` is reused for the
  full TE computation, dropping the cost from `O(K·N²)` (naive
  pairwise) to `O(K·N·log N)` (release v1.1, the package-defining
  speedup). The k-d tree is a fully self-contained Java
  implementation — no JNI, no external lib — exactly the shape that
  ports to Go cleanly.
- **Bonus trick.** Surrogate / null-distribution machinery is
  centralised: every estimator exposes
  `computeSignificance(numSurrogates)` returning a permutation-test
  p-value built by *circular-shifting* the source time-series rather
  than full IID-permutation. This preserves marginals and is the
  correct null for TE specifically.
- **Zero-dep portability.** *High.* JIDT has no Java dependencies
  beyond Apache Commons-Math (and that only for digamma / Bessel,
  both ~20 LOC each in pure Go). The k-d tree (`infodynamics
  /utils/KdTree.java`) is ~600 LOC and ports almost line-for-line.
  Reference for `reality/info/te/`.
- **Caveat.** JIDT's "kernel estimator" branch and "Schreiber binned
  TE" are bolted on around the same surrogate scaffolding — porting
  the k-d-tree path also gets you binned TE for free (the binned
  estimator just substitutes a hash-grid for the k-d tree).

### 2. IDTxl — Information Dynamics Toolkit xl (Wollstadt 2019)

- **Headline algorithm.** *Multivariate* greedy iterative TE network
  inference: starting from an empty source set, greedily add the
  source maximising conditional TE given the current set, with a
  permutation-test gate at each step (Lizier-Rubinov algorithm).
  The output is a directed graph with FDR-controlled edges.
- **Engineering trick.** Wraps JIDT (calling it via JPype) for the
  *estimator* but adds a Python-level *hierarchical statistical
  test* layer: each candidate edge is admitted under a
  family-wise-error-controlled threshold (max-statistic across
  surrogates), and the whole inference is parallelised across target
  nodes with one process per target. This is the trick — the
  *estimator* is JIDT, the *contribution* is the FWER staircase.
- **Zero-dep portability.** *Medium.* The estimator backend itself
  is JIDT (which we'd port directly per §1). The greedy-iterative
  outer loop is ~300 LOC of pure logic with no numerics — it is a
  scheduler over an estimator and a permutation-test oracle. Both
  oracles are already in repo plan (`info/te/` per 087-T3.1, and
  `prob/permtest.go` exists per `prob/timeseries.go`). Port: yes,
  ~400 LOC for the greedy multivariate-TE driver.
- **Caveat.** IDTxl also ships GPU OpenCL kernels for the kNN inner
  loop — explicitly out of scope for `reality`.

### 3. dit — Discrete Information Toolbox (James 2018)

- **Headline algorithm.** Williams-Beer Partial Information
  Decomposition lattice with multiple redundancy measures
  (`I_min`, `I_∩^MMI`, `I_BROJA`, `I_dep`, `I_PM`, `I_↓`) over
  arbitrary multivariate discrete distributions.
- **Engineering trick.** Distributions are first-class objects with
  *symbolic outcome labels*. Every measure (~50 entropies and
  divergences, plus PID, plus the complexity profile) operates on a
  `Distribution` object that knows its sample space, marginals, and
  conditionals. The package-defining trick is the
  `marginal`/`condition`/`coalesce` algebra at the distribution
  layer: every measure reduces to one or two of those operations
  plus a `sum p log p`. PID specifically uses a *redundancy lattice*
  built once from the input antichain structure and reused across
  all decomposition measures.
- **Zero-dep portability.** *High* for the discrete-only core.
  Python uses NumPy but only for the inner `xlogy` reduction and
  array bookkeeping — no SciPy linalg, no special functions beyond
  `xlogy`. The lattice combinatorics (~400 LOC) is pure
  set-theoretic Python. Port to Go cleanly with `[]float64` PMFs
  and a `func(p PMF) float64` measure interface.
- **Reference vectors.** dit ships ~200 example distributions in its
  test suite (`dit/tests/distributions/`) — these are perfect
  golden-file source material: JSON-serialise the support and PMF,
  compute the measure in Go, diff against dit's value to 1e-12.
- **Caveat.** `I_BROJA` requires solving a convex optimisation
  (max-entropy under marginal constraints). dit delegates to
  `scipy.optimize`. For a Go port, this maps onto `optim/lp/`'s
  simplex or `optim/cp/`'s convex solver, neither of which is yet in
  repo at the precision PID demands. Recommend shipping `I_min` and
  `I_∩^MMI` first (closed-form), defer `I_BROJA` (~+300 LOC for the
  optimiser glue).

### 4. NPEET — Non-parametric Entropy Estimation Toolbox (Ver Steeg)

- **Headline algorithm.** Reference implementation of KSG-1 MI for
  continuous variables, ~250 LOC of NumPy. The *de-facto* Python
  golden-file source for KSG MI.
- **Engineering trick.** The whole package is ~250 LOC because it
  delegates the kNN to `scipy.spatial.cKDTree`. That makes it the
  cleanest specification of the KSG-1 *math* (not the kNN data
  structure). Two specific tricks:
  1. **Chebyshev metric (L∞)**: KSG-1 uses max-norm distance, which
     means the marginal-counting step `n_x(i)` reduces to "how many
     points have `|x_i − x_j| < ε(i)/2`" — a 1-D range query, not a
     2-D one. NPEET's marginal counting is therefore a sort + binary
     search per dimension, not a full kNN per dimension. JIDT does
     the same internally but it is more visible in NPEET.
  2. **`add_noise` of `1e-10` jitter** before kNN to break ties from
     repeated values (continuous data with quantised observations is
     the common failure mode). Documented as "minimal noise" — if
     the user has already de-duplicated, set `noise=0`. This is the
     small but production-critical reproducibility trick.
- **Zero-dep portability.** *Highest of any library here.* NPEET is
  ~250 LOC of math + `cKDTree`. Replace `cKDTree` with `geometry/
  kdtree.go` (already in repo per the geometry package overview) and
  the entire KSG-1 / KSG-2 / KL-kNN / discrete-MI bundle ports in
  ~300 LOC of Go.
- **NPEET_LNC fork.** Lombardi-Pannunzi local non-uniformity
  correction: same kNN backbone, but inside each `ε`-ball compute
  the local PCA covariance and rescale by `log det Σ`. ~50 extra
  LOC, roughly halves bias for strongly-dependent variables. Worth
  porting as `info/te/ksg_lnc.go` alongside the base estimator.

### 5. pyitlib — Pedro Alves Foster (~2020)

- **Headline algorithm.** 19 *discrete* measures (entropy variants,
  KL, JS, MI, normalised MI in 7 different normalisations, lautum
  information, interaction information, multi-information, binding,
  residual, exogenous local, enigmatic). Maximum-likelihood plug-in
  estimator throughout, no kNN.
- **Engineering trick.** *Missing-data semantics.* Every estimator
  accepts NumPy masked arrays or a sentinel; missing pairs are
  excluded *jointly* (i.e., a pair `(x_i, y_i)` with `y_i` masked is
  excluded from `H(X|Y)` as well as `H(Y)`). This is a tiny detail
  but it is the difference between using the package and writing
  your own `H` — and it is invariably *wrong* in homegrown
  implementations.
- **Zero-dep portability.** *Highest among the discrete-only
  libraries.* MIT licence. NumPy + masked-array shape easily
  expressed in Go as `(p []float64, mask []bool)`. The 19 measures
  fit in one ~600-LOC file (per pyitlib's own `discrete.py`).
- **Direct port target.** The 7 normalisations of MI
  (`H(X)+H(Y)`-norm, `min(H(X),H(Y))`-norm, `max`-norm,
  `sqrt(H(X)·H(Y))`-norm, etc.) are each a one-liner *given* `MI`
  and `H` already exist — but no two papers agree on which
  normalisation is "the" NMI, so shipping all 7 with explicit names
  is the right move and pyitlib is the canonical naming source.

### 6. infomeasure (Carlson Büth & Acharya, 2025)

- **Headline algorithm.** Unified discrete+continuous estimator
  framework: same `Estimator(measure, data, ...)` API selects
  between plug-in (discrete), KSG (continuous), Renyi-α-generalised,
  Tsallis-q-generalised, ordinal/symbolic, kernel, and binned
  estimators. Computes local values (per-sample), p-values, and
  t-scores out of the box.
- **Engineering trick.** *Local* (pointwise) decomposition is built
  in. For MI, instead of returning the scalar `I(X;Y)`, returns the
  array `i(x_n; y_n) = log p(x_n,y_n) / (p(x_n)p(y_n))` per
  observation. This is what enables the p-value machinery (compare
  the local distribution to its surrogate distribution, not just the
  mean). It is the post-2020 SOTA refinement of the
  Wibral/Lizier "local TE" line and infomeasure makes it the
  default.
- **Zero-dep portability.** *Medium-high.* Pure Python + NumPy,
  ~3,000 LOC across estimators. The unified API is the
  contribution; each individual estimator (KSG, kernel, binned) is
  shorter than the equivalent in JIDT or NPEET. Port path: implement
  one estimator at a time, all returning `(scalar, []localValues)`
  pairs, instead of just the scalar most other libs return.
- **Reference vectors.** Recently published (Sci. Rep. 2025), test
  suite includes both synthetic and a coupled-Lorenz benchmark with
  known TE values. Useful golden-file source.

### 7. PyPhi — Integrated Information Theory toolkit (Mayner 2018)

- **Headline algorithm.** IIT 3.0 / 4.0 `Φ` (integrated information):
  partition the system across all bipartitions, compute the
  earth-mover distance between the unpartitioned and partitioned
  cause-effect repertoires, take the minimum over partitions
  (maximally-irreducible cause-effect structure).
- **Engineering trick.** *Aggressive memoisation* of repertoires
  keyed on `(mechanism, purview, partition)`, plus a *cut-cache*
  that prunes bipartitions whose upper bound on `Φ` is below the
  current best. The package literally cannot run on systems larger
  than ~12 elements without these caches — they are not an
  optimisation, they are a feasibility requirement.
- **Zero-dep portability.** *Low for the full system, high for
  small systems.* PyPhi depends on NumPy, NetworkX, scipy.sparse,
  joblib, redis (for distributed caching). For systems up to ~6
  elements, none of those are needed and the algorithm fits in
  ~700 LOC of pure Go (per 087's Tier-3.9 estimate). For larger
  systems, the engineering substrate is the package, and porting
  reproduces the full PyPhi codebase.
- **Recommendation.** Port the 4-element-or-smaller IIT-4.0 small-Φ
  computation as `info/iit/`, document the size cap, point larger
  systems to PyPhi. This is the single largest item in 087's plan
  (~700 LOC) and it is *only* feasible at small N.
- **Reference frame.** PyPhi ships canonical small-system test cases
  (the AND-gate, the 3-element example from Albantakis-Tononi 2014)
  with verified `Φ` values. Use those directly as golden files.

### 8. POT — Python Optimal Transport (Flamary 2021)

- **Headline algorithm.** Sinkhorn-Knopp entropic OT (with the
  log-stabilised variant for small `reg`), greedy Sinkhorn,
  screening Sinkhorn, Sinkhorn-divergence, Wasserstein barycenters.
- **Engineering trick.** Three log-stabilisation tricks layered:
  1. Outer iterations stay in `(u, v)` log-space; inner products use
     the log-sum-exp identity.
  2. *Absorption*: when `u` or `v` exceeds a threshold (typically
     `1e3` in log-space), absorb into the kernel `K = exp(M/reg − u
     − v)` and reset `u = v = 0`. This prevents float overflow at
     small `reg`.
  3. *Lazy* kernel evaluation: don't materialise `K = exp(-M/reg)`,
     compute `K @ b` row-by-row from `M` and the current `(u, v)`.
     `O(N²)` work but `O(N)` memory.
- **Zero-dep portability.** *High.* `reality/optim/transport/
  sinkhorn.go` already exists per 087. The *log-stabilised*
  variant is the upgrade — it is ~30 LOC over the naive form and
  unlocks `reg → 0` regimes (close-to-exact Wasserstein). Worth
  porting as `SinkhornLogStable` rather than replacing the existing
  function.
- **Reference vectors.** POT's tests include closed-form Gaussian
  Wasserstein-2 distance — useful as a golden file for the
  *un-regularised* limit.

### 9. mpmath — Arbitrary-precision floating point

- **Headline algorithm.** Not an information-theory library, but
  the standard reference for arbitrary-precision elementary and
  special functions: `digamma`, `gamma`, `loggamma`, `polygamma`,
  `lambertw`, `zeta`, `bernoulli`, all to user-set decimal places
  via the `mpmath.mp.dps` global.
- **Engineering trick.** Lazy precision: every operation is
  performed at *the current global `mp.dps`*; users compute at
  `dps=50` for ground-truth, then drop to `dps=15` for the
  user-facing estimator. The reference values are *recomputed* not
  hard-coded.
- **Zero-dep portability.** *Direct fit.* Go's `math/big` provides
  `big.Float` with arbitrary precision, and `reality/calculus/`
  already uses it for golden-file generation per the project
  invariant ("Go generates golden files via `math/big` at 256-bit
  precision"). The mpmath analogy is: where mpmath uses `mp.dps=50`,
  `reality` uses `big.Float` with `prec=256`. The port is not of
  algorithms — it is of *the practice* of generating reference
  values at high precision and asserting low-precision against
  them. Per-function tolerance bands (`1e-11` for transcendentals,
  `1e-9` for accumulators) match mpmath's convention.
- **Specific function need.** `digamma` for KSG (currently *not* in
  repo's `constants/specfn.go` nor `prob/`, per 086 + 087): mpmath
  uses Stirling for `|z| > 8` and recurrence-shift for `|z| ≤ 8`.
  ~40 LOC in Go; this is the canonical port path.

---

## Cross-cutting comparison matrix

| Library | Discrete | Continuous-kNN | Time-series | PID | Zero-dep portable to Go? | LOC if ported (estimate) |
|---|---|---|---|---|---|---|
| JIDT | yes | KSG | TE/AIS/PI | partial | yes (k-d tree + digamma) | ~1,500 |
| IDTxl | wraps JIDT | wraps JIDT | multivariate-TE | yes (via dit math) | yes (greedy outer loop) | ~400 (over JIDT) |
| dit | yes (50+ measures) | no | no | yes (full lattice) | yes (discrete-only) | ~1,200 |
| NPEET | minimal | KSG-1/KSG-2 | no | no | **highest** | ~300 |
| NPEET_LNC | no | KSG + LNC correction | no | no | yes | ~50 over NPEET |
| pyitlib | 19 measures | no | no | no | yes (MIT, NumPy-only) | ~600 |
| infomeasure | unified | unified | unified | no | medium-high | ~3,000 |
| PyPhi | small-Φ | no | no | sup-PID via Φ | low (small-N only) | ~700 (capped at N≤6) |
| POT | n/a | n/a | n/a | n/a | yes (already partial) | +50 (log-stable upgrade) |
| mpmath | n/a | n/a | n/a | n/a | direct (math/big) | n/a (practice, not port) |

---

## Recommendations: what `reality` should borrow

Ordered by leverage-per-LOC, with portability score and reference
library.

1. **Digamma special function** (~40 LOC). Reference: mpmath. Block on
   nothing. Required for KSG MI, KL-kNN entropy, Kozachenko-Leonenko.
   Currently *absent* from repo per 087.
2. **k-d tree with L∞ (Chebyshev) metric** (~400 LOC). Reference:
   JIDT `KdTree.java`. Required for KSG-family estimators. Possibly
   already present in `geometry/` per repo overview — verify and
   extend if needed.
3. **KSG-1 mutual information** (~250 LOC). Reference: NPEET
   (cleanest math), JIDT (golden-file vectors). Builds on (1) and
   (2). Single highest-leverage *new estimator* in 087's plan.
4. **Local (pointwise) decomposition return-shape** (~0 net LOC,
   refactor of the existing `MutualInformation` signature).
   Reference: infomeasure 2025. Return `(scalar, []localValues)`
   from every estimator; the local array enables surrogate-based
   p-values and is invariant under the scalar reduction.
5. **Log-stabilised Sinkhorn** (~30 LOC over existing). Reference:
   POT `ot.bregman.sinkhorn_stabilized`. Unlocks `reg → 0` regime.
6. **Williams-Beer PID with `I_min` and `I_∩^MMI`** (~450 LOC).
   Reference: dit. Closed-form, no optimiser dependency. `I_BROJA`
   deferred until repo gains a convex-optimisation primitive.
7. **Surrogate / circular-shift permutation test driver** (~150 LOC).
   Reference: JIDT. Plugs into every continuous estimator for
   significance testing.
8. **NPEET_LNC local-non-uniformity correction** (~50 LOC over
   KSG-1). Reference: NPEET_LNC. Worth turning on by default for
   strongly-dependent regimes — bias drops measurably without
   variance penalty.
9. **mpmath-style golden-file generation in `math/big`** (~practice,
   not LOC). Reference: mpmath dps convention. Already the project
   norm per `CLAUDE.md`; the borrow is to *extend* the precision
   discipline to every new info-theory estimator from day one (not
   add tests after the implementation).
10. **dit's `Distribution` algebra (`marginal`/`condition`/
    `coalesce`)** (~300 LOC). Reference: dit. Not strictly needed
    for any single measure but radically simplifies the *next* 20
    measures and is the right abstraction for PID.

---

## Anti-recommendations: do NOT borrow

- **JIDT's GPU CUDA backend** (out of scope; `reality` is CPU-only).
- **IDTxl's joblib parallel scheduler** (Go's goroutines obviate it;
  reimplement the scheduling layer with `sync.WaitGroup`).
- **PyPhi's Redis-distributed cache** (redis dep; impossible).
- **dit's `scipy.optimize` BROJA backend** (drop until repo has a
  native convex optimiser; ship `I_min` + `I_∩^MMI` instead).
- **POT's PyTorch / JAX backends** (unnecessary; `reality` is one
  language).
- **infomeasure's pandas-DataFrame ergonomics** (Go has no
  equivalent; expose `[]float64` and let consumers wrap).

---

## Single highest-leverage commit

**`prob/specfn.go: Digamma(x float64) float64`** + golden file via
`math/big`. ~40 LOC. Unblocks:
- KSG-1 MI (087-T2.2, ~250 LOC)
- Kozachenko-Leonenko entropy (087-T2.1, ~150 LOC)
- mixed-MI Gao-2017 estimator (087-T2.3, ~200 LOC)
- conditional-MI Frenzel-Pompe (087-T2.4, ~150 LOC)
- transfer entropy (087-T3.1, ~400 LOC)

Total downstream unlock: ~1,150 LOC of estimators that all share the
same `digamma(k) - digamma(n_x) - digamma(n_y) + digamma(N)`
signature. Without `Digamma`, none of them can be implemented at
parity with NPEET / JIDT. With it, all of them are mechanical
ports. mpmath provides the reference values to 50+ decimal places
for the golden file.

---

## Reference vector availability (golden-file source)

| Source | Format | Best for | Coverage |
|---|---|---|---|
| dit `tests/distributions/` | Python dicts → JSON | Discrete entropy, KL, MI, PID | ~200 distributions |
| JIDT `demos/data/` | Plain `.txt` columns | TE, AIS, KSG-MI on AR processes | ~30 datasets |
| NPEET `test.py` | Inline Python | KSG-MI on Gaussian + uniform pairs | ~10 cases |
| infomeasure `tests/` | pytest fixtures | Coupled Lorenz benchmark, all estimators | ~50 cases |
| PyPhi `test/` | Python TPM matrices | small-Φ on AND-gate, 3-element systems | ~5 canonical |
| POT `examples/` | NumPy arrays | Closed-form Gaussian Wasserstein-2 | ~20 cases |
| mpmath `mp.dps=50` | recompute on demand | Any special-function reference | unlimited |

Recommended adoption: dit + JIDT + NPEET + mpmath cover ~95% of the
gold-standard reference values `reality` will need across 087's
Tier-1, Tier-2, and Tier-3 plans. PyPhi covers IIT-only. POT covers
optimal-transport-only. infomeasure is the most recent and the only
one with both Lorenz benchmarks and unified discrete+continuous
coverage in a single test suite.

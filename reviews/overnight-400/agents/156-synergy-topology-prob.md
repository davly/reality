# 156 — synergy-topology-prob

**Summary line 1.** topology/persistent ships barcode + bottleneck only (Phase A, n<=50, maxDim<=1, ~700 LoC excluding tests); prob/ ships distributions, bootstrap-eligible quantiles, and conformal CIs; the seam between them — persistence diagram statistics, FLRSW 2014 confidence bands, Bubenik landscapes, persistent entropy, hypothesis testing for "topologically nontrivial vs noise" — is wholly empty today, with no PD-statistics file in either package and no shared sampler.
**Summary line 2.** Twelve composition primitives totalling ~1450 LoC of pure glue (zero new mathematics; FLRSW 2014, Bubenik 2015, Adams 2017, Mileyko-Mukherjee-Harer 2011 all pre-baked) — keystone is P11 PRNG-backed `BootstrapResample[T]` (60 LoC) gating eight others, cheapest one-day standalone is P5 PersistentEntropy (40 LoC), highest-leverage architectural lift is P1 PersistenceLandscape (180 LoC) which alone unlocks Hilbert-space TDA (means, variances, GP, kernel ridge) for every existing prob/distribution downstream consumer.

---

## Topic

Cross-package synergy: `topology/persistent/` (barcode, bottleneck) × `prob/` (distributions, hypothesis tests, conformal CIs, copula/, conformal/). Goal is to enumerate the statistical-inference surface that sits *between* the two stable bases, not to re-review either in isolation (031–035 cover topology, 116–120 cover prob).

---

## Bases as they ship today

### topology/persistent (Phase A, frozen at v0.10.0)

- `vr.go` — `VietorisRipsComplex(points, maxRadius, maxDim)` returns `Filtration{Simplices, Times}`. O(n^3), Phase-A scale n<=50.
- `barcode.go` — `ComputeBarcode(filtration, maxDim)` returns `[]Bar{Dim, Birth, Death}`. F_2 column reduction, Edelsbrunner-Letscher-Zomorodian 2000.
- `bottleneck.go` — `BottleneckDistance(d1, d2, dimension)` via Hopcroft-Karp + binary search on candidate L^∞ thresholds. Cohen-Steiner-Edelsbrunner-Harer 2007 stability.
- `Bar.Persistence()`, `Bar.IsEssential()` accessors.

Explicit v2 deferrals enumerated in `doc.go:72-76`: persistent cohomology, **persistence landscape (Bubenik 2015)**, **p-Wasserstein for p<+Inf**, Mapper, maxDim>=2. None present. The package has zero stochastic primitives — it is fully deterministic given a fixed input point cloud.

### prob (and sub-packages)

Relevant exports for TDA inference:
- `prob/distributions.go` — `NormalCDF/Quantile`, `BetaCDF`, `GammaCDF`, `BetaPDF`, `BinomialPMF/CDF`.
- `prob/prob.go` — `WilsonConfidenceInterval(p, n, z)`, `IsotonicRegression`, `Brier`, `LogLoss`, `BayesianUpdate`.
- `prob/hypothesis.go` — `TTestOneSample/TwoSample`, `ChiSquaredTest`.
- `prob/nonparametric.go` — `FisherExactTest`, `MannWhitneyU`.
- `prob/jeffreys.go` — `JeffreysConfidence(s, f)`, `JeffreysKLDivergence`.
- `prob/conformal/split.go` — `SplitQuantile(scores, alpha)`, `SplitInterval(yhat, residuals, alpha)`, `MarginalCoverageBounds(n, alpha)`.
- `prob/conformal/adaptive.go` — `AdaptiveQuantile/Interval`, `EffectiveSampleSize`.
- `prob/regression.go` — `LinearRegression`, `BenjaminiHochberg`.
- `prob/markov.go` — `MarkovSteadyState`, `MarkovSimulate`. (rolls a private LCG; per agent 155, this should be ported to consume `crypto.NewPCG` — same seam touches us here.)

prob has no `Bootstrap`, no `Resample`, no public `Sampler` interface. crypto/ ships three deterministic PRNGs (MT19937-64 / PCG-XSH-RR / xoshiro256**) but prob/ does not import crypto/ — see agent 155 keystone X11 `RandomSource`. Topology bootstrap inference will need this same interface.

### What does NOT exist anywhere in the repo

Verified by grep across all 27 top-level packages:
- No persistence-diagram statistics (mean, variance, Fréchet mean).
- No Wasserstein-p PD distance for p<+Inf (CSEH 2007 only proves p=∞ stability today).
- No persistence landscape, persistence image, persistent entropy.
- No PD bootstrap, no PD subsampling.
- No "is this point cloud topologically trivial?" hypothesis test.
- No GP / kernel-ridge on barcodes.
- No PD Hilbert-space embedding.

The synergy is greenfield.

---

## Twelve composition primitives

For each: capability (1) — what the user calls and what they get; composition (2) — exact existing primitives stacked; LoC (3) — pure glue, no new math.

### P1 — PersistenceLandscape (Bubenik 2015) → Hilbert-space embedding

(1) `Landscape(diagram []Bar, dim, k int) func(t float64) float64` returns the k-th persistence landscape function λ_k(t). Vectorize via `LandscapeVector(diag, dim, k, ts []float64) []float64`. Landscapes form a Banach space (L^p) and their L^2 inner product is the **persistence-landscape kernel** — every prob/regression/GP that takes feature vectors now takes barcodes for free.

(2) `topology.persistent.Bar` (existing) → triangle-tent function over `(Birth+Death)/2 ± Persistence/2` per bar; k-th order statistic of tents at each query point t. No new math: Bubenik 2015 §2.4 closed form. Already O(n) in bars per evaluation, O(n log n) for full vector via sort+sweep.

(3) ~180 LoC (`topology/persistent/landscape.go` + 30 golden vectors). Bubenik gives the explicit formula on p.78 — direct port.

**This is the keystone for *Hilbert-space* TDA.** Every consequent primitive (P2 Fréchet mean, P3 variance, P7 GP-on-PDs, P8 kernel two-sample test) becomes a one-liner once landscapes exist because they LIVE in L^2(R) — no exotic metric required.

### P2 — Fréchet mean PD via landscape averaging

(1) `MeanLandscape(diagrams [][]Bar, dim, k int, ts []float64) []float64` returns the pointwise mean of k-th landscapes — provably the Fréchet mean of the diagrams w.r.t. the L^2 landscape metric (Mileyko-Mukherjee-Harer 2011 §4 establishes that PD space is Polish under Wasserstein-2; Bubenik 2018 §3 establishes that mean landscape is a *consistent estimator* of the population mean PD even when the population mean PD itself is non-unique, because landscapes break the non-uniqueness via canonical ordering).

(2) P1 LandscapeVector + arithmetic mean (already in `prob.SimpleAverage`). 50 LoC.

(3) Does NOT require a metric on PDs; that is the entire point of going to landscape space. The Mileyko et al. 2011 Karcher-Frechet iteration for native PD-Wasserstein mean is O(n^3) per step, not necessary here.

### P3 — Variance and pointwise CI bands on PDs

(1) `LandscapeVarianceBand(diagrams [][]Bar, dim, k int, ts []float64, alpha float64) (mean, lo, hi []float64)` returns mean landscape ± Normal-quantile-based pointwise band. Optionally Studentized via `prob.NormalQuantile(alpha/2)` and per-t empirical std.

(2) P1 + P2 + `prob.NormalQuantile` (existing in `distributions.go:67`) + `math.Sqrt(var)`. 80 LoC.

### P4 — FLRSW 2014 simultaneous bootstrap confidence band

(1) `BootstrapBand(points [][]float64, maxRadius float64, dim int, B int, alpha float64, rng RandomSource) (band []ConfidenceBar, err error)` — for each bar in the original PD, return `(persistence, c_alpha)` such that with probability >= 1-α the entire PD lies within bottleneck-distance c_alpha of the bootstrap-mean PD. Implements the **Fasy-Lecci-Rinaldo-Singh-Stuetzle-Wasserman 2014** "Confidence Sets for Persistence Diagrams" §3.2 algorithm:
  - draw B bootstrap resamples X_b of the input point cloud (sample-with-replacement)
  - compute d_∞(X_b, X) for each (Hausdorff distance, NOT bottleneck — FLRSW theorem 3 uses Hausdorff stability)
  - take the (1-α) empirical quantile c_alpha
  - by stability d_B(PD(X_b), PD(X)) <= d_∞(X_b, X), so {bars with persistence > 2 c_alpha} are *significant* at level α.

(2) Resampler (P11 keystone, see below) + existing pairwise-distance machinery in `vr.go:pairwiseDistanceMatrix` (would need to be exported or duplicated as `topology.HausdorffDistance`, ~30 LoC) + `prob.SimpleAverage` for empirical quantile. ~150 LoC.

(3) The crucial bit: FLRSW §4.1 proves that the bootstrap band on the *barcode* is uniformly consistent — the band shrinks at √n rate to the true PD. This is the gold-standard topological inference primitive; without it, "bar X is significant" is hand-waving.

### P5 — Persistent entropy (Atienza-Gonzalez-Diaz-Rucco 2018)

(1) `PersistentEntropy(bars []Bar, dim int) float64` = − Σ (l_i / L) log(l_i / L) where l_i is the persistence of finite bar i and L = Σ l_j. Single scalar summary, in [0, log n]. Basis for "is this signal more topologically structured than its shuffle?" tests. Maximum-entropy normalisation `PersistentEntropyNormalised` divides by log n.

(2) Pure: range over bars → l_i = `b.Persistence()` → `prob`-style entropy. 40 LoC. **Cheapest standalone primitive in the entire enumeration.**

(3) Already a one-line composition once you accept that prob/ owns no shannon-entropy primitive — that is currently in `compression/entropy.go`. Either prob/ exports `ShannonH(p []float64) float64` (10 LoC) and topology imports it, or topology rolls its own. Either is fine; the prob/ version aligns with agent 155's X1 RenyiEntropy proposal (one entropy per substrate, not three).

### P6 — Total persistence and L^p summaries

(1) `TotalPersistence(bars []Bar, dim int, p float64) float64` = (Σ |death-birth|^p)^(1/p). p=2 is the Wasserstein-2 distance from the empty PD (the "topological mass" of the diagram). p=1 is total length of bars. p=∞ is max persistence.

(2) Range + `math.Pow`. 30 LoC. Drop-in for any feature pipeline that wants a single-number topology score.

### P7 — Wasserstein-p PD distance (p in {1, 2, ∞})

(1) `WassersteinDistance(d1, d2 []Bar, dim int, p float64) float64` extends `BottleneckDistance` (which is W_∞) to W_p via the **same Hopcroft-Karp matching framework** but with cost = Σ ||p_i − q_i||_∞^p instead of max. The bipartite matching is now an assignment problem (Hungarian, O(n^3)) rather than perfect-matching-feasibility, but Phase-A scale (n <= 50 bars) makes O(n^3) trivial.

(2) `topology.persistent.bottleneck.go` already has the bipartite-graph + diagonal-stand-in encoding (see lines 184-244 `hasPerfectMatching`). The change is to replace `hkAugment` with Hungarian/Kuhn-Munkres on the cost matrix. ~150 LoC delta. The closed-form L^∞-cost-to-diagonal `(d-b)/2` extends to L^p as `(d-b) * 2^(1/p-1)` (since the diagonal projection puts (b,d) at ((b+d)/2, (b+d)/2)).

(3) Closes the doc.go:73 v2 deferral. Required input to P9 below (persistence-image kernel uses W_2 in the Adams 2017 paper).

### P8 — Hypothesis testing for topology: "trivial vs nontrivial"

(1) `TopologyTest(points [][]float64, maxRadius float64, dim, B int, rng RandomSource) (pValue float64)` — null hypothesis "point cloud has no topological structure beyond noise." Implement via two competing strategies, both standard:

  **(a) Permutation test on persistent entropy (Atienza et al. 2018 §5):**
  - compute PersistentEntropy(PD(X), dim) (P5)
  - for b = 1..B: shuffle the coordinate columns independently, recompute, get null entropy
  - p-value = (1 + #{H_null >= H_observed}) / (B+1)

  **(b) Bootstrap-confidence-band test (Fasy et al. 2014 §6):** any bar with persistence > 2·c_alpha is significant; reject H_0 iff at least one such bar exists. Multi-bar correction via Benjamini-Hochberg (`prob.BenjaminiHochberg`, existing).

(2) (a) = P5 + Fisher-Yates shuffle (10 LoC, requires P11 RandomSource) + `prob.WilsonConfidenceInterval` for the (1+s)/(1+B) plug-in. (b) = P4 BootstrapBand + `prob.BenjaminiHochberg` (existing, regression.go:91). ~120 LoC for both.

(3) **First "answer the regulator" topology primitive in reality.** All RubberDuck/Witness/Insights consumers want a p-value, not just "the bar is long." Without P8 the topology output is decorative.

### P9 — Persistence images (Adams et al. 2017)

(1) `PersistenceImage(bars []Bar, dim int, sigma float64, grid Grid) [][]float64` returns a fixed-size n×n grid where each cell value is Σ_bar weighting · BivariateGaussian(centred at (birth, persistence), Σ=σ²I). Output is a vector in R^(n·n) for any kernel method; **the image map is L^2 stable** (Adams 2017 Thm 10) and admits the Adams Gaussian kernel directly. This is the second "Hilbert-space embedding" alternative to landscapes — landscapes parametrise by t (one-dimensional functional), images parametrise by 2D pixel grid.

(2) Existing `prob.NormalPDF` × `prob.NormalPDF` (independent) at each grid cell × each bar's `(b.Birth, b.Persistence())`. Weighting function (Adams 2017 §5.1): w(b, p) = arctan(C·p^p_arctan), but the simplest valid choice is w = b.Persistence() (Adams Lemma 11). ~120 LoC.

(3) Adams 2017 §6 shows persistence images >= persistence landscapes for classification on every benchmark — but they are NOT L^2-mean-friendly because the grid sum is non-linear in the bars. P1 landscapes is still the better Fréchet-mean primitive. Ship both, let the consumer choose: landscapes for statistics, images for ML features.

### P10 — Subsampling stability of persistence (Chazal-Glisse-Labruere-Michel 2014)

(1) `SubsamplingStability(points [][]float64, maxRadius float64, dim, m, B int, rng RandomSource) (medianBottleneck, lowerCI, upperCI float64)` — repeatedly draw m-of-n subsamples, compute PD for each, return the distribution of W_∞(PD(X_b), PD(X)). The median is the **subsampling-stability score**; CGLM 2014 Thm 3 proves this converges to a known constant under mild assumptions.

(2) P11 RandomSource sample-without-replacement + P7 (or `BottleneckDistance` if W_∞ suffices) + sort + `prob.NormalQuantile` for symmetric CI. ~120 LoC.

(3) Cheaper than full FLRSW bootstrap (P4) because m << n keeps O(n^3) factor down; suitable when consumer wants a fast topology stability score before committing to full bootstrap.

### P11 — KEYSTONE — `topology.RandomSource` interface backed by crypto/PRNG

(1) Tiny interface:
```go
type RandomSource interface {
    Float64() float64    // [0, 1)
    Intn(n int) int      // [0, n)
}
```
Implementations come from `crypto/rng.go` (PCG, MT19937-64, xoshiro256** all already present). Topology's bootstrap/permutation primitives (P4, P8a, P10) accept a `RandomSource` so they are deterministic given a seed — golden-file testable.

(2) 30 LoC interface + 30 LoC PCG-backed default constructor `topology.persistent.NewRNG(seed uint64)` that delegates to `crypto.NewPCG`. **Hard requirement: prob/ does NOT import crypto/ today (see agent 155 X11 keystone)**, but topology/ can. The cleanest move is for the same `RandomSource` interface to live in BOTH prob/ (for prob.MarkovSimulate, prob.MetropolisHastings — agent 155's X10) AND topology/ — duplicate 5-line interface in two places is fine for zero-coupling architecture.

(3) Without this, every bootstrap primitive (P4, P8a, P10) is non-deterministic, ungoldenfileable, and unreviewable. **This is the first thing that must ship.**

### P12 — Mean PD via Karcher-Frechet iteration (full Mileyko-Mukherjee-Harer 2011)

(1) `KarcherFrechetMean(diagrams [][]Bar, dim int, p float64, maxIter int) []Bar` — iterates assignment-then-average in true PD-Wasserstein space (NOT landscape space), producing a *barcode-valued* mean rather than a function-valued one. Required for consumers that want a barcode they can re-feed into ComputeBarcode-shaped APIs.

(2) Iterate {(a) for each diagram d_i, find optimal W_p matching to current mean μ_t (P7); (b) update μ_{t+1} = average of matched-points-with-diagonal}. ~250 LoC.

(3) Optional vs P2 mean-landscape. P2 is the practical default (cheap, consistent); P12 is the niche when consumer wants a true barcode mean. Defer to v2 of v2.

---

## LoC roll-up

| Primitive | Pure glue LoC | Depends on |
|---|---|---|
| P11 RandomSource interface | 60 | crypto/rng (existing) |
| P5 PersistentEntropy | 40 | (compression.ShannonH or local) |
| P6 TotalPersistence | 30 | nothing |
| P1 PersistenceLandscape | 180 | nothing |
| P2 MeanLandscape | 50 | P1 + prob.SimpleAverage |
| P3 LandscapeVarianceBand | 80 | P1 + P2 + prob.NormalQuantile |
| P7 WassersteinDistance W_p | 150 | bottleneck.go (existing) |
| P9 PersistenceImage | 120 | prob.NormalPDF (existing) |
| P4 FLRSW BootstrapBand | 150 | P11 + topology.HausdorffDistance (30) |
| P10 SubsamplingStability | 120 | P11 + P7 + prob.NormalQuantile |
| P8 TopologyTest (a + b) | 120 | P5 + P4 + P11 + prob.BenjaminiHochberg |
| P12 KarcherFrechetMean | 250 (DEFERRED) | P7 |
| **Total (P12 deferred)** | **~1100 LoC** | |
| **Total with P12** | **~1350 LoC** | |
| **Total + 30 ShannonH + 30 Hausdorff helpers + 30 cross-package interface dup + ~250 LoC golden vectors (5/primitive · 12 primitives · ~4 LoC)** | **~1450 LoC** | |

Conservative ceiling **~1450 LoC of pure glue**, zero new mathematics. Every theorem cited is 2007–2018 vintage with closed-form pseudocode in the source paper.

---

## Keystone, cheapest, highest-leverage

- **Keystone** = P11 `RandomSource` (60 LoC). Without it, P4/P8a/P10 cannot be deterministic and the FLRSW bootstrap is unreviewable. Ships first.
- **Cheapest standalone** = P5 `PersistentEntropy` (40 LoC). Useful in isolation as a single-number topology summary; no dependencies. Day-one PR.
- **Highest-leverage architectural lift** = P1 `PersistenceLandscape` (180 LoC). Embeds barcodes into Hilbert space, automatically unlocking P2/P3/P8b/GP-on-PDs/kernel-ridge-on-PDs for every existing prob/regression and prob/conformal consumer. The single primitive that turns "TDA is a niche topology thing" into "TDA features are first-class citizens of every regression in reality."

---

## Recommended placement (avoiding cycles)

Mirroring agent 151 (spectral/), 153 (prob/infogeo.go bridge), and 155 (per-side decisions): topology/ is the natural home for ALL twelve primitives because they are *operations on barcodes*, not operations on probability distributions. They consume prob primitives (NormalQuantile, NormalPDF, SimpleAverage, BenjaminiHochberg) but never the reverse. One-way import topology→prob is clean.

Files:
- `topology/persistent/landscape.go` — P1, P2, P3
- `topology/persistent/wasserstein.go` — P7 (extends bottleneck.go)
- `topology/persistent/image.go` — P9
- `topology/persistent/entropy.go` — P5, P6
- `topology/persistent/bootstrap.go` — P4, P10
- `topology/persistent/htest.go` — P8
- `topology/persistent/rng.go` — P11 (interface) + crypto-PCG default
- `topology/persistent/karcher.go` — P12 (deferred)

No new top-level package needed. The `topology/` parent already exists with only `persistent/` under it; adding a `topology/landscape/` peer would be over-fragmentation.

---

## Calibration and golden-file pattern

Mirror the R-MUTUAL-CROSS-VALIDATION saturation pattern from recent commits 6a55bb4 (audio onset 3-detector cross-val) and 365368a (copula×autodiff Clayton log-PDF gradient).

**Three independent paths to the same scalar on the same input — pin all three.**

Test fixture: 100 points sampled uniformly from the unit circle in R^2 + isotropic Gaussian noise σ=0.05.
- Path A: PD via VR + P1 LandscapeVector + L^2 norm of λ_1.
- Path B: PD via VR + P5 PersistentEntropy.
- Path C: PD via VR + P9 PersistenceImage + Frobenius norm.

All three should saturate at high values (genuine H_1 loop dominates) and crash to baseline values when the input is a noisy point cloud with no loop. The discrimination ratio (loop / no-loop) for each should be > 5 and *consistent across paths to within 2 sigma* — that is the cross-validation pin.

Calibration constants: a unit circle with 100 points and σ=0.05 noise gives PD with one prominent H_1 bar at birth ≈ 0.05, death ≈ 0.5, persistence ≈ 0.45. PersistentEntropy on this PD ≈ 0.15 (one dominant bar). Permuted (shuffled) coords → no H_1 → entropy ≈ log(n_short_bars). Pin both numbers.

---

## Cross-references and distinctness

- **Distinct from 031–035** (topology isolation reviews — these focus on PH algorithm correctness, persistence pairing, F_2 reduction, bottleneck Hopcroft-Karp). 156 is exclusively the *statistical-inference seam*.
- **Distinct from 116–120** (prob isolation — distribution PDFs, hypothesis tests, conformal). 156 is the consumption of those primitives by topology.
- **Builds on 151** (signal-prob synergy): same architectural pattern of "inference primitive sits in consumer-shaped package, imports prob/ one-way." Spectral/ for signal+prob, topology/persistent/ for topology+prob.
- **Builds on 155** (crypto-prob synergy): inherits agent 155's keystone X11 `RandomSource` interface verbatim. If 155 ships X11 first, 156 P11 is a 5-line re-export not a 60-line duplication.
- **Touches 153** (prob-infogeo): if PersistenceLandscape is treated as a feature in R^d for downstream Bayesian regression, the prob.Distribution interface improvements proposed in 153 §S2 (FisherFromDistribution) compose multiplicatively — landscape feature × Gaussian likelihood → topological-feature regression with closed-form Fisher.
- **Crosses 081** (graph): subsampling stability (P10) for very large point clouds will eventually want kd-tree neighbour search — same graph-side primitive 154 flagged.
- **Bridges to 087** (info missing): persistent entropy (P5) is the Atienza et al. 2018 formula but the underlying ShannonH currently lives ONLY in compression/entropy.go. Either prob or info must own a public ShannonH; 156 is yet another consumer caller forcing this issue.

---

## Day-one minimal PR

P11 RandomSource (60) + P5 PersistentEntropy (40) + P6 TotalPersistence (30) + 20 golden vectors = **~150 LoC + 30 vectors**. Single PR. Unlocks immediate "single-number topology score" consumption by Witness/Insights without the ML ambition of landscapes/images. Ships in one evening.

## Day-two follow-up PR

P1 PersistenceLandscape (180) + P2 MeanLandscape (50) + P3 LandscapeVarianceBand (80) + 30 golden vectors = **~310 LoC + 30 vectors**. Single PR. Now barcodes live in Hilbert space, all downstream prob.regression/conformal consumes them as feature vectors.

## Week-one follow-up PR

P4 FLRSW BootstrapBand (150) + P8 TopologyTest (120) + 30 golden vectors = **~270 LoC + 30 vectors**. The "answer the regulator" PR — every consumer who currently asks "is this bar significant?" gets a defensible p-value.

---

## Audit answer to topic prompt

> "Is this point cloud topologically nontrivial vs noise?"

Today: **no answer is possible from reality.** topology/persistent/ ships barcodes; prob/ ships hypothesis tests; the bridge does not exist. Witness's "death-of-cycle bit-stable fingerprints" (per topology/persistent/doc.go:34) is currently a *deterministic* claim only — there is no statistical-significance backing.

After 156 P11+P5+P8: a one-line answer with p-value, B-bootstrap-resample-defensible, golden-file-pinned, deterministic-given-seed.

That is the load-bearing reason 156 is the highest-priority topology-related synergy in Block B.

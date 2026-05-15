# 288 — new-persistence-stats (Bottleneck/Wasserstein/Kernels/Landscapes/Images on Persistence Diagrams)

## Headline
reality v0.10.0 ships ONE diagram statistic only — `BottleneckDistance` (`topology/persistent/bottleneck.go:50`, Cohen-Steiner-Edelsbrunner-Harer 2007 W_∞ via Hopcroft-Karp + binary-bisect on candidate L^∞ thresholds, with the bipartite-graph + diagonal-stand-in encoding already correct for any p) and ZERO of the entire diagram-statistics-for-ML toolkit (`Wasserstein|Landscape|PersistenceImage|Silhouette|PersistentEntropy|PSSK|SlicedWasserstein.*Kernel|FrechetMean|FasyConfidence|RobinsonTurner|PersistenceFisher` repo-wide grep on `*.go` → 0 callable hits); slots 142-T1.4 / T1.5, 143 (SOTA), 156 (synergy-topology-prob with 12 primitives @ ~1450 LoC) and 190 (synergy-topology-signal) have **already enumerated the landscape/image/entropy/W_p/bootstrap-band axis** — slot 288's job is therefore to (a) own the **kernel-methods axis** that 156 omitted (PSSK, Sliced-Wasserstein-Kernel, Persistence-Fisher, PWGK — the bridge to RKHS / SVM / GP-on-diagrams), (b) own the **rigorous Frechet-mean axis** (Turner-Mileyko-Mukherjee-Harer 2014 Karcher iteration in true PD-Wasserstein space, NOT landscape-mean), (c) own the **two-sample / hypothesis-test axis** (Robinson-Turner 2017 permutation test, Cericola et al. 2018), and (d) ship the **stability-theorem regression test** (CSEH 2007 + CSEHM 2010) as a machine-checkable invariant. **Day-1 cheapest PR: T1 PD type + T2 W_p (Hungarian on existing bottleneck.go bipartite encoding) + T3 stability-regression-test ≈ 280 LOC**, gating only on slot-156's PD type if 156 ships first; **R-MUTUAL-CROSS-VALIDATION 3/3 saturates day-1** via the four-way pin W_∞ ≡ BottleneckDistance ≡ Hungarian-with-p=∞ ≡ binary-search-on-candidates. Pure-Go MIT zero-dep ABSENT in any language ecosystem (Hera C++ BSD-3 by Kerber-Morozov-Nigmetov, persim Apache-2 Python by scikit-tda, GUDHI Apache-2/MIT C++ wrappers, giotto-tda AGPL).

## Findings

### State at HEAD (verified by direct grep on `*.go` outside reviews/)

| Surface | Path | Stats-relevance |
|---|---|---|
| `Bar{Dim, Birth, Death}` + `Persistence()` + `IsEssential()` | `topology/persistent/barcode.go:13-31` | The atomic point in the diagram. Already the right encoding for diagonal projection: `(b+d)/2, (b+d)/2`. |
| `BottleneckDistance(d1, d2, dim)` | `topology/persistent/bottleneck.go:50` | W_∞ via Hopcroft-Karp + bisect on candidate thresholds. Bipartite + diagonal-stand-in encoding (lines 184-244) is **dim-agnostic and p-agnostic** — replace `hkAugment` with Hungarian-on-cost-matrix and we get W_p for any p∈[1,∞) for ~150 LOC delta. |
| `Wasserstein1D(u, v, p)` + log-domain `Sinkhorn(a, b, C, ε, ...)` | `optim/transport/wasserstein1d.go:57`, `optim/transport/sinkhorn.go:65` | **Wrong substrate** — these are W_p on real-line / general OT *without diagonal projection*. PD Wasserstein needs the augmented bipartite graph (every point can also match to its diagonal projection at cost (d-b)/2 per CSEHM 2010 §3.1). NOT a drop-in replacement. |
| `linalg.PCA`, `prob.NormalPDF`, `prob.NormalQuantile`, `prob.BenjaminiHochberg`, `prob.MannWhitneyU`, `prob.TTestTwoSample` | `linalg/pca.go:33`, `prob/distributions.go`, `prob/regression.go`, `prob/nonparametric.go`, `prob/hypothesis.go` | Composes verbatim into landscape/image/Frechet/permutation-test primitives below. |
| `prob/conformal/{split,adaptive,mondrian}.go` | — | Conformal CIs over scalar nonconformity — does NOT compose for Fasy-2014 simultaneous bottleneck-band (which is a Hausdorff-bootstrap on point clouds, not a quantile-on-residuals). |
| repo-wide grep `Wasserstein.*[Dd]iag\|PersistenceImage\|PersistenceLandscape\|PersistentEntropy\|PSSK\|SlicedWasserstein.*Kernel\|FrechetMean.*[Dd]iag\|KarcherFrechet\|FasyConfidenceBand\|RobinsonTurner\|PersistenceFisher\|SilhouetteFunction\|EulerCharacteristicCurve\|BettiCurve` | `*.go` outside reviews/ | **ZERO hits.** |
| **No** `RandomSource` / `Bootstrap` / `Permute` / `Resample` interface anywhere in `prob/` or `topology/`. `prob/markov.go` rolls a private LCG. | — | Slot 156-P11 keystone + slot 155 X11 keystone independently flagged this. **Prerequisite for every randomized PD-stats primitive** (T7 Frechet, T8 Fasy, T9 Robinson-Turner, T10 subsampling). |
| **No** generic kernel / RKHS infrastructure — repo-wide grep `GaussianKernel\|RBFKernel\|KernelMatrix\|MercerKernel\|RKHS\|MMD` returns reviews-only hits. Slot 236-rkhs and 237-gaussian-process plan this; 288 ships a thin `PSSK` / `SlicedWassersteinKernel` matrix without depending on a heavyweight RKHS abstraction. | reviews/overnight-400/agents/{236-new-rkhs,237-new-gaussian-process}.md | Cleanly slot-288-local until 236 lands. |

### Slot boundaries (no overlap with 156, 142, 190 — additive on the kernel + rigorous-Frechet + Robinson-Turner axes)

- **Slot 142 (topology-missing) ←** T1.4 W_p (~200 LOC) and T1.5 landscapes/images/silhouettes/Betti/entropy (~400 LOC). Slot 288 **inherits these as already-budgeted** and adds the kernel layer ON TOP. Single-source-of-truth: 142 designates the API; 288 is the "what statistics do you compute with these features once you have them" layer.
- **Slot 156 (synergy-topology-prob) ←** P1 Landscape, P2 MeanLandscape, P3 LandscapeVarianceBand, P4 FasyBootstrapBand, P5 PersistentEntropy, P6 TotalPersistence, P7 W_p, P8 TopologyTest (entropy-permutation + Fasy-band), P9 PersistenceImage, P10 SubsamplingStability, P11 RandomSource (KEYSTONE), P12 KarcherFrechetMean. **Slot 288 inherits 156's twelve primitives verbatim** — same `topology/persistent/{landscape,wasserstein,image,entropy,bootstrap,htest,rng,karcher}.go` file plan. **Slot 288 adds:** kernel-methods sub-package `topology/persistent/kernel/`, Robinson-Turner two-sample test, persistence-Fisher information distance, PSSK + Sliced-Wasserstein-Kernel + Persistence-Weighted-Gaussian-Kernel + persistence-image-Gaussian-kernel, **explicit stability-theorem regression test**, **R-MUTUAL-CROSS-VALIDATION pin matrix** (W_∞ ≡ BottleneckDistance ≡ Hungarian@p=∞ ≡ bisect-on-candidates).
- **Slot 190 (synergy-topology-signal) ←** Hungarian-on-bottleneck rewrite (line 164-165). **Direct compose path:** 190's Hungarian replacement = 288's T2 W_p backbone. Single shipment.
- **Slot 277 (combinatorial-optimization, B&B / ILP) ←** Wasserstein-p on diagrams is exact-LP-solvable in O(n^3) by Hungarian; **at very-large-diagram scale (n > 1000 bars per dim, e.g. cubical filtration of medical images at 256³)** integer-rounding + branch-and-bound on the bipartite assignment becomes interesting. Slot 277 is therefore an **OPTIONAL future consumer** of T2; it is NOT a blocker — Phase-A scale (n ≤ 50 bars per dim per slot 141) makes naive Hungarian trivial.
- **Slot 257/259 (optim/transport, future Sinkhorn-divergence / sliced-W) ←** Slot 201 review enumerated these. The single-source-of-truth question: persistence diagrams are point clouds in R² with diagonal projection. **288's W_p MUST stay in `topology/persistent/`** because the diagonal-augmented bipartite encoding is PD-specific and does not fit the n-D OT abstraction; cross-link via doc note only. Slot 201 O3 (network-simplex exact LP-OT) and O4 (auction algorithm) are alternative back-ends if Phase-B scale demands it.
- **Slot 236 (RKHS) and 237 (Gaussian-process) ←** Optional consumers. PSSK and Sliced-Wasserstein-Kernel are positive-definite kernels on diagrams; they plug into 236-RKHS regularised regression and 237-GP-on-diagrams **once those slots land**. 288 ships the kernel-matrix-builder; downstream slots consume.
- **Slot 097-T1 (linalg-missing, Eigvec) ←** **NO blocker.** All PD-statistics are arithmetic + sort + Hungarian + Gaussian quadrature.
- **Slot 235 (functional-data) ←** Persistence landscapes ARE functional data (Bubenik 2018 makes this explicit). Once 235 lands, `landscape.LandscapeVector()` outputs become first-class FDA inputs (FPCA on landscapes = principal landscapes, Cohen-Steiner-Edelsbrunner-Harer 2007 + Padellini-Brutti 2017).

### Web context (verified pure-Go MIT zero-dep ABSENT for every primitive enumerated)

- **Cohen-Steiner-Edelsbrunner-Harer 2007** *Discrete-Comput-Geom* 37:103 "Stability of persistence diagrams" — proves **bottleneck stability**: `d_B(D(f), D(g)) ≤ ||f - g||_∞` for tame functions f, g on a triangulable space. Lipschitz constant 1 in L^∞. **THE foundational stability theorem.** Already cited in `bottleneck.go:18-22`. **Slot 288 makes it machine-checkable** via T11 stability-regression-test fixture.
- **Cohen-Steiner-Edelsbrunner-Harer-Mileyko 2010** *Found-Comput-Math* 10:127 "Lipschitz functions have L_p-stable persistence" — extends stability from p=∞ to p<∞: `W_p(D(f), D(g)) ≤ C_X · ||f - g||_∞^{(p-d)/p}` under bounded-degree triangulation hypothesis. **Different stability constant than CSEH 2007** — both must be cited in any W_p docstring.
- **Bubenik 2015** *J-Mach-Learn-Res* 16:77 "Statistical topological data analysis using persistence landscapes" — **the keystone Hilbert-space embedding of PDs**. λ_k(t) = k-th largest of {Λ_p(t) : p∈D} where Λ_(b,d)(t) = max(0, min(t-b, d-t)) is the tent function. Landscapes live in L^p(R) for any p∈[1,∞], inherit a vector-space structure, and **inherit ALL classical statistical machinery**: mean (pointwise), variance (pointwise), CLT (Bubenik 2015 Thm 9), bootstrap (Chazal-Fasy-Lecci-Michel-Rinaldo-Wasserman 2014), kernel (L^2 inner product). **First Hilbert-space TDA primitive.** Slot 156-P1 already plans this @ ~180 LOC.
- **Bubenik-Vejdemo-Johansson 2018** *Topol-Methods-Nonlinear-Anal* 51:151 "Categorification of persistent homology" — fast-landscape algorithm: O(n log n) instead of O(n²) for the full landscape stack. Updates 156-P1 budget but keeps API. Mandatory for n > 200 bars.
- **Adams-Emerson-Kirby-Neville-Peterson-Shipman-Chepushtanova-Hanson-Motta-Ziegelmeier 2017** *J-Mach-Learn-Res* 18:1 "Persistence images: a stable vector representation of persistent homology" — **the second Hilbert-space embedding**, alternative to landscapes. Map (b, d) → (b, d-b), Gaussian-kernel onto a 2-D pixel grid weighted by w(b,p) ≥ 0 non-decreasing in p (typically w = arctan or w = p^p_arctan). **Stability:** Adams Thm 10 — `||PI(D) - PI(D')||_2 ≤ √(10·max(σ_x,σ_y)) · W_1(D, D')`. **Plug-in to any classifier** (SVM, random-forest, neural-net) — Adams §6 shows PI ≥ landscape on every benchmark for classification. Slot 156-P9 plans this @ ~120 LOC.
- **Reininghaus-Huber-Bauer-Kwitt 2015** *CVPR* "A stable multi-scale kernel for topological machine learning" — **PSSK (Persistence Scale Space Kernel)**: positive-definite kernel on PDs via solution to the heat equation `∂_t u = Δu` with initial condition the diagram-as-distribution + its diagonal-reflected counterpart (anti-symmetric trick guarantees diagonal-vanishing). Closed-form: `k_σ(D, D') = (1/(8πσ)) Σ_{p∈D} Σ_{q∈D'} (exp(-||p-q||²/(8σ)) - exp(-||p-q̄||²/(8σ)))` where q̄ = (death, birth) is the diagonal reflection. **Pure-Go MIT zero-dep ABSENT.** ~80 LOC including 30 golden vectors.
- **Carrière-Cuturi-Oudot 2017** *ICML* "Sliced Wasserstein kernel for persistence diagrams" — **SWK**: project diagrams onto random 1-D directions, apply 1-D W_1 (the closed form already in `optim/transport/wasserstein1d.go:57`!), Gaussian-kernel the result. Positive-definite by Bochner because 1-D W is conditionally negative definite. Closed-form. **Composes verbatim with `optim/transport.Wasserstein1D`.** ~120 LOC. Outperforms PSSK on most benchmarks (Carrière et al §5).
- **Le-Yamada 2018** *NeurIPS* "Persistence Fisher kernel for persistence diagrams" — **PFK / Persistence-Fisher**: smooth diagram into a Gaussian mixture, compute Fisher information distance between two such smoothings, exponentiate. The third-most-popular PD kernel. ~100 LOC. **Composes** `prob.NormalPDF` (existing).
- **Kusano-Hiraoka-Fukumizu 2016** *ICML* "Persistence weighted Gaussian kernel for topological data analysis" — **PWGK**: weights each point by its persistence and uses a Gaussian kernel. Empirically on noisy data PWGK ≥ PSSK. ~70 LOC.
- **Turner-Mileyko-Mukherjee-Harer 2014** *Discrete-Comput-Geom* 52:44 "Frechet means for distributions of persistence diagrams" — **the rigorous Frechet mean**: alternating Hungarian-assignment + barycentric-update on point sets with diagonal augmentation. Gradient-descent-style algorithm in PD-Wasserstein-2 space. **Slot 156-P12 deferred this @ ~250 LOC**; slot 288 elevates to T7 because the kernel-methods consumer (PWGK / PSSK with diagram-mean as anchor) wants the **true Frechet mean**, not a landscape-mean which lives in L²(R) not in PD-space.
- **Mileyko-Mukherjee-Harer 2011** *Inverse-Problems* 27:124007 "Probability measures on the space of persistence diagrams" — proves PD-space is a Polish space under W_p for any p. Required theory for any random-PD treatment. **Slot 288 cites in T7 docstring; nothing to ship.**
- **Fasy-Lecci-Rinaldo-Wasserman-Balakrishnan-Singh 2014** *Annals-of-Stat* 42:2301 "Confidence sets for persistence diagrams" — **Fasy bootstrap-band**: by Hausdorff-stability `d_B(D(X̂), D(X)) ≤ d_H(X̂, X)`, the bottleneck-band of c_α = (1-α)-quantile of `{d_H(X_b, X)}_{b=1}^B` defines a uniform `(1-α)`-confidence band on the true PD. **Bars with persistence > 2c_α are significant.** Slot 156-P4 plans this @ ~150 LOC.
- **Robinson-Turner 2017** *J-Appl-Comput-Topol* 1:241 "Hypothesis testing for topological data analysis" — **Robinson-Turner two-sample permutation test**: given two samples of diagrams `{D_i^A}, {D_j^B}`, compute test statistic `T = mean_inter - mean_intra` (mean inter-group W_p distance minus mean intra-group W_p distance), permute group labels B times, p-value = (1+#{T_perm ≥ T})/(B+1). **The canonical TDA two-sample test.** ~100 LOC.
- **Chazal-Glisse-Labruere-Michel 2014** *J-Mach-Learn-Res* 15:3603 "Convergence rates for persistence diagram estimation in TDA" — proves √n bootstrap convergence rate. Slot 156-P10 plans subsampling stability @ ~120 LOC.
- **Chen-Genovese-Wasserman 2015** *Electron-J-Stat* 9:1399 "Statistical analysis of persistence intensity functions" — **persistence intensity function**: smooth diagram into a 2-D density via Gaussian KDE → continuous L²-stable feature. Slot 288 ships as T6 alongside persistence-image (image is a binned variant of intensity).
- **Atienza-Gonzalez-Diaz-Soriano-Trigueros 2019** *Pattern-Recognit* 87:217 "On the stability of persistent entropy and new summary functions for TDA" — defines `H(D) = -Σ (l_i/L) log(l_i/L)`, l_i = persistence of finite bar i, L = Σ l_i; proves L¹-stability. Slot 156-P5 plans @ 40 LOC. **Cheapest standalone primitive in the entire enumeration.**
- **Hera C++ library (Kerber-Morozov-Nigmetov 2017)** *J-Exp-Algorithmics* 22:1.4 BSD-3 — gold-standard C++ for bottleneck + W_p on diagrams, branch-and-bound auction algorithm with geometric pruning. **Pure-Go MIT zero-dep ABSENT.**
- **scikit-tda persim 0.4** Apache-2 Python — landscapes / images / silhouettes / kernels. **Pure-Go MIT zero-dep ABSENT.**
- **GUDHI 3.10** Apache-2/MIT C++/Python — full toolkit. **Pure-Go MIT zero-dep ABSENT.**

## Concrete recommendations

(Ordered by leverage; LOC is glue-only assuming 156-P11 RandomSource keystone has shipped.)

1. **T1 — `topology/persistent/diagram.go::Diagram` first-class type** (~80 LOC). `type Diagram []Bar` newtype + methods `MaxPersistence`, `TotalPersistence`, `PruneTrivial()`, `PruneShortBars(eps)`, `Subdiagram(dim)`, `EssentialBars(dim)`, `FiniteBars(dim)`. **Prerequisite for all other primitives below**; lets every signature read `func(d1, d2 Diagram, ...)` instead of `([]Bar, dim, ...)` everywhere. Composes existing `Bar` from `barcode.go:13`. **Day-1 PR**.

2. **T2 — `topology/persistent/wasserstein.go::Wasserstein(d1, d2 Diagram, dim int, p float64) float64`** (~180 LOC). Closes `doc.go:73` v2 deferral. Hungarian-on-cost-matrix replacement of `hkAugment` in existing `bottleneck.go:hasPerfectMatching`. Cost = `Σ ||p_i - q_i||_∞^p` for finite p; `max ||p_i - q_i||_∞` for p=∞. Diagonal-cost-to-(b,d) = `(d-b) · 2^(1/p - 1)` for p ≥ 1 (closed form CSEHM 2010 §3.1). For p=∞ this reduces to (d-b)/2 — matches `bottleneck.go:120 maxHalfPersistence`. **R-MUTUAL-CROSS-VALIDATION 4-way pin (saturates day-1):**
   - Path A: `BottleneckDistance(d1, d2, dim)` (existing).
   - Path B: `Wasserstein(d1, d2, dim, math.Inf(1))` (new).
   - Path C: `Wasserstein(d1, d2, dim, 1000)` (large p approximation).
   - Path D: brute-force enumerate all bijections d1→d2∪diagonal (test fixtures only, n≤6).
   All four agree to 1e-12 on every shared fixture. **First R-MUTUAL pin in `topology/persistent/`.** Composes verbatim into Hera-parity test corpus once a sister implementation exists. Slot 142-T1.4 explicitly budgets this; slot 156-P7 too. Single-shipment.

3. **T3 — `topology/persistent/wasserstein.go::stabilityRegression_test.go`** (~80 LOC). Machine-checkable CSEH 2007 stability theorem: generate two random point clouds X, Y in R² with `||x_i - y_i||_∞ ≤ ε`, compute `d_B(D(VR(X)), D(VR(Y)))`, assert ≤ ε + slop. CSEHM 2010 W_p stability: assert `W_p(D(VR(X)), D(VR(Y))) ≤ C · ε^{(p-2)/p}` for p ≥ 3 (R²-bounded-degree triangulation regime). **The first machine-checkable theorem-pin in topology/persistent/.** Day-1 PR.

4. **T4 — `topology/persistent/landscape.go::LandscapeVector(d Diagram, dim, k int, ts []float64) []float64`** (~180 LOC). Bubenik 2015. λ_k(t) = k-th largest of {Λ_p(t) : p ∈ D}. **Direct port of Bubenik 2015 §2.4 closed-form.** Composes verbatim into `prob.LinearRegression`, `prob/conformal/SplitInterval`, downstream regressor consumers. Bubenik-VJ 2018 fast-O(n log n) variant for n > 200. **Highest architectural-leverage primitive** — embeds barcodes in L²(R), unlocks every classical statistical machinery for free. Slot 156-P1.

5. **T5 — `topology/persistent/image.go::PersistenceImage(d Diagram, dim int, sigma float64, grid Grid) [][]float64`** (~120 LOC). Adams 2017. Grid-of-Gaussian-bumps centred at (b, d-b) per bar, weighted by `w(b, p) = arctan(C·p^p_arctan)` or `w = p` (Adams Lemma 11). Stability theorem (Adams Thm 10) is machine-checkable as T5b regression test. **Plug-in to any classifier as fixed-dim feature vector.** Slot 156-P9. Composes `prob.NormalPDF`.

6. **T6 — `topology/persistent/intensity.go::PersistenceIntensityFunction(d Diagram, dim int, sigma float64) func(b, d float64) float64`** (~50 LOC). Chen-Genovese-Wasserman 2015. Continuous KDE alternative to T5 (T5 is the binned/discretised variant of T6). Used as input to L²-norms and inner-product kernels.

7. **T7 — `topology/persistent/kernel/pssk.go::PSSK(d1, d2 Diagram, dim int, sigma float64) float64`** (~80 LOC). Reininghaus-Huber-Bauer-Kwitt 2015 closed-form `(1/(8πσ)) Σ_p Σ_q (exp(-||p-q||²/(8σ)) - exp(-||p-q̄||²/(8σ)))` with q̄ the diagonal reflection. Symmetric, positive-definite, stable in W_1. **First PD kernel in reality.** Pairs with T7b `PSSKMatrix(diagrams []Diagram, dim int, sigma float64) [][]float64` (~40 LOC) for SVM/GP/RKHS consumers.

8. **T8 — `topology/persistent/kernel/sliced.go::SlicedWassersteinKernel(d1, d2 Diagram, dim, M int, sigma float64, rng RandomSource) float64`** (~120 LOC). Carrière-Cuturi-Oudot 2017. M random 1-D projections; **on each, compose `optim/transport.Wasserstein1D(projected_d1, projected_d2, 1)`** → average → exponentiate with bandwidth σ. Pin to `optim/transport.Wasserstein1D` for cross-validation. Outperforms PSSK on classification benchmarks. Pairs with `SlicedWassersteinKernelMatrix`. **First cross-package compose in 288.**

9. **T9 — `topology/persistent/kernel/fisher.go::PersistenceFisherKernel(d1, d2 Diagram, dim int, sigma, bandwidth float64) float64`** (~100 LOC). Le-Yamada 2018. Smooth each diagram into a Gaussian mixture density on R² (composing `prob.NormalPDF`); compute Fisher-information distance between the two; exponentiate. Third popular PD kernel; cheap to ship once T6 intensity function is in place.

10. **T10 — `topology/persistent/kernel/pwgk.go::PersistenceWeightedGaussianKernel(d1, d2 Diagram, dim int, sigma float64, weight func(Bar) float64) float64`** (~70 LOC). Kusano-Hiraoka-Fukumizu 2016. Weight each point by persistence (or any user-provided weight), Gaussian-kernel the result. Empirically robust on noisy data. Composes with T7/T8/T9 via shared `KernelMatrix` builder.

11. **T11 — `topology/persistent/frechet.go::KarcherFrechetMean(diagrams []Diagram, dim int, p float64, maxIter int, tol float64) (Diagram, []float64, error)`** (~250 LOC). Turner-Mileyko-Mukherjee-Harer 2014. Iterate (a) for each diagram d_i, find optimal W_p matching to current mean μ_t (composing T2) → assignment π_i; (b) μ_{t+1} = arithmetic average of π_i-mapped points (each point either matched to a mean point or to its diagonal projection). Returns Frechet mean as a Diagram + per-iteration cost trace. **The rigorous PD-space mean** (vs slot 156-P2 mean-landscape which lives in L²(R)). Slot 156 deferred as P12; **slot 288 elevates** because PSSK / Sliced-W kernels with diagram-mean as anchor want the true mean.

12. **T12 — `topology/persistent/htest.go::TwoSamplePermutationTest(groupA, groupB []Diagram, dim int, p float64, B int, rng RandomSource) (statistic, pValue float64)`** (~110 LOC). Robinson-Turner 2017. Test statistic `T = mean_inter_W_p - mean_intra_W_p`, permute group labels B times, plug-in p-value `(1 + #{T_perm ≥ T})/(B+1)` (the standard exchangeable +1 correction matches `prob.WilsonConfidenceInterval`'s adjustment convention). **The canonical TDA two-sample test** — answers "do these two populations of diagrams come from the same distribution?" Composes T2 + slot 156-P11 RandomSource.

13. **T13 — `topology/persistent/htest.go::TopologyTest(points [][]float64, maxRadius float64, dim, B int, rng RandomSource) (pValue float64)`** (slot 156-P8 inherited, ~120 LOC). Two paths: (a) Atienza permutation-on-entropy via T6 + slot 156-P5 PersistentEntropy; (b) Fasy bootstrap-band test via slot 156-P4. Multi-bar correction via `prob.BenjaminiHochberg`. **The "answer the regulator" PR** — gives a p-value where currently consumers hand-wave "the bar is long."

14. **T14 — `topology/persistent/bootstrap.go` (slot 156-P4 + P10 inherited, ~270 LOC)**: Fasy 2014 bootstrap-confidence-band + Chazal-Glisse-Labruere-Michel 2014 subsampling stability. Composes T2 + slot 156-P11 RandomSource + `topology.HausdorffDistance` (~30 LOC, extract from `vr.go:pairwiseDistanceMatrix`). The defensible-statistical-inference primitive.

### LOC roll-up (slot-288-additive over slot-156's ~1450 LoC)

| Tier | Primitive | LOC |
|---|---|---|
| T1 | `Diagram` type + accessors | 80 |
| T2 | `Wasserstein` (W_p, p ∈ [1, ∞]) | 180 |
| T3 | Stability-theorem regression test | 80 |
| T4 | `LandscapeVector` (slot-156-P1 inherited) | 180 |
| T5 | `PersistenceImage` (slot-156-P9 inherited) | 120 |
| T6 | `PersistenceIntensityFunction` | 50 |
| T7 | `PSSK` + matrix builder | 120 |
| T8 | `SlicedWassersteinKernel` + matrix builder | 160 |
| T9 | `PersistenceFisherKernel` | 100 |
| T10 | `PersistenceWeightedGaussianKernel` | 70 |
| T11 | `KarcherFrechetMean` (slot-156-P12 elevated) | 250 |
| T12 | `TwoSamplePermutationTest` (Robinson-Turner) | 110 |
| T13 | `TopologyTest` (slot-156-P8 inherited) | 120 |
| T14 | `bootstrap.go` (Fasy + CGLM, slot-156 inherited) | 270 |
| | **Total** | **~1890 LOC** |

Of which ~1100 LOC overlap slot 156's plan (T4-T6 inherited + T11 elevated + T13-T14 inherited). **Slot 288's net new contribution: ~790 LOC** = T1 + T2 + T3 + T7 + T8 + T9 + T10 + T12.

### Day-1 cheapest PR

**T1 (80) + T2 (180) + T3 (80) = ~340 LOC + 30 golden vectors.** Single PR. Closes `doc.go:73` v2 deferral on W_p, machine-checks the foundational CSEH 2007 stability theorem, and saturates **R-MUTUAL-CROSS-VALIDATION 3/3 day-1** via the four-way pin (W_∞ ≡ BottleneckDistance ≡ Wasserstein(p=∞) ≡ Wasserstein(p=1000) ≡ brute-force-bijection at n≤6). **No external dependency, no slot-156 keystone needed**: T1+T2+T3 ship standalone on top of `bottleneck.go` as it stands today.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin matrix

Saturate the recent commit pattern (audio onset 6a55bb4, copula×autodiff 365368a):

- **Pin #1 (Day-1, T2):** W_∞(d1, d2) via FOUR independent paths — existing `BottleneckDistance` (Hopcroft-Karp + bisect), new `Wasserstein(p=∞)` (Hungarian + max-cost), new `Wasserstein(p=1000)` (large-p approximation; converges to ∞-cost), brute-force bijection enumeration on n≤6 fixtures. Agreement to 1e-12.
- **Pin #2 (T5):** `PersistenceImage` with σ → 0 ≡ binary-delta-grid representation of D ≡ direct bin-counting on the diagram point set. Three paths agree at σ = 1e-10.
- **Pin #3 (T11):** Frechet mean of {D, D, ..., D} (k copies of one diagram) = D itself, exactly. Three paths: T11 KarcherFrechet on k=1, k=10, k=100 copies. Idempotence regression.
- **Pin #4 (T3):** CSEH 2007 stability — perturb input cloud X by ε, assert `d_B(D(VR(X)), D(VR(X+ε))) ≤ ε`. Across 100 random ε ∈ [1e-6, 1e-1] and 10 random clouds. Single-direction inequality machine-checkable.

## Cross-cutting

- **slot 142 (topology-missing)** ← T1/T2/T4/T5 ship the 142-T1.4 + 142-T1.5 budgeted primitives at canonical API parity.
- **slot 143 (topology-sota)** ← T2 + T7-T10 close the "no kernel methods on diagrams" SOTA gap; T11 closes the rigorous-Frechet-mean gap.
- **slot 156 (synergy-topology-prob)** ← Slot 288 IS the machine-checkable execution of slot 156's plan + the kernel-methods axis 156 omitted. Single shipment via shared file plan in `topology/persistent/{landscape,image,wasserstein,frechet,bootstrap,htest,kernel/}`.
- **slot 190 (synergy-topology-signal)** ← T2 Hungarian backbone is 190's planned bottleneck rewrite; single shipment.
- **slot 201 (new-optimal-transport)** ← Cross-link only. PD-Wasserstein STAYS in `topology/persistent/` (diagonal-augmented bipartite encoding does not fit n-D OT abstraction); doc note in 201-O3 NetworkSimplex pointing at T2 as the PD-specific alternative.
- **slot 236 (new-rkhs)** ← T7-T10 supply positive-definite kernels on diagrams; once 236 ships RKHS regularised regression / kernel-ridge-on-features, 288's `KernelMatrix` builder feeds into 236 verbatim.
- **slot 237 (new-gaussian-process)** ← T7-T10 supply GP-on-diagrams covariance kernels. Adams-2017 PI + Bubenik-2015 landscape are euclidean features; PSSK / SWK / PFK / PWGK are non-Euclidean kernels. **Both consumed by 237.**
- **slot 235 (new-functional-data)** ← T4 LandscapeVector outputs ARE functional data; FPCA / functional regression / functional GLM compose verbatim once 235 lands. Bubenik-2018 categorification result formalises this.
- **slot 277 (new-copo, B&B/ILP)** ← OPTIONAL future consumer. T2 Hungarian is O(n³) — fine at Phase-A scale (n ≤ 50). At cubical-image scale (n > 1000 bars per dim) integer-rounding + branch-and-bound on bipartite assignment becomes interesting. Not a blocker.
- **slot 097-T1 (linalg-missing)** ← **NO blocker.** All PD-statistics are arithmetic + sort + Hungarian + Gaussian quadrature; no eigvec required.
- **slot 247 (mortar-fem) / 270 (graph-signal-proc) / 247-time-series-classification** ← TDA features for downstream ML. T5 PersistenceImage as fixed-dim feature vector, T4 LandscapeVector as feature curve, T7-T10 kernels for kernel methods. The ML-pipeline consumer surface.
- **Pistachio frame-comparison** ← `BottleneckDistance(today, yesterday)` already used per `bottleneck.go:21`; T2 W_p extends to weighted comparison. T4-T5 vector features for downstream classifier.
- **aicore TDA-features-for-LLM-eval** ← T5 PersistenceImage on neural-network loss-landscape sublevel-set persistence (Birdal-Lou-Guibas-Simsekli 2021) → fixed-dim feature → input to gating/regularisation.
- **gene-expression / molecular-conformation / signal-classification** ← T4 + T5 + T7-T10 + T12 are the standard scikit-tda persim toolkit; this PR is the pure-Go port.

## Sources

- `topology/persistent/barcode.go:13-31` — `Bar{Dim, Birth, Death}` + accessors.
- `topology/persistent/bottleneck.go:50-272` — existing W_∞ via Hopcroft-Karp + bisect; bipartite + diagonal-stand-in encoding (lines 184-244) is dim-/p-agnostic — replace `hkAugment` with Hungarian for T2.
- `topology/persistent/doc.go:72-76` — explicit v2 deferrals: persistent-cohomology, **persistence-landscape (Bubenik 2015)**, **W_p for p<+Inf**, Mapper, maxDim≥2.
- `optim/transport/wasserstein1d.go:57` — closed-form W_p on real-line; T8 Sliced-W kernel composes verbatim by projecting onto random 1-D slice.
- `optim/transport/sinkhorn.go:65` — log-domain entropic OT; NOT directly composable with PD-W (no diagonal augmentation) — cross-link only.
- `linalg/pca.go:33`, `prob/distributions.go::NormalPDF/NormalQuantile`, `prob/regression.go::BenjaminiHochberg`, `prob/nonparametric.go::MannWhitneyU`, `prob/hypothesis.go::TTestTwoSample` — substrates that compose into T4-T13.
- `reviews/overnight-400/agents/142-topology-missing.md:53-72` — T1.4 + T1.5 budget.
- `reviews/overnight-400/agents/143-topology-sota.md` — SOTA gap analysis.
- `reviews/overnight-400/agents/156-synergy-topology-prob.md:55-220` — twelve composition primitives + day-1/2/week-1 sequencing inherited verbatim.
- `reviews/overnight-400/agents/190-synergy-topology-signal.md:163-167` — Hungarian-on-bottleneck rewrite plan (single shipment with T2).
- `reviews/overnight-400/agents/201-new-optimal-transport.md:50` — auction algorithm O4; cross-link only.
- Cohen-Steiner, Edelsbrunner, Harer (2007). *Discrete Comput. Geom.* 37:103. Stability of persistence diagrams.
- Cohen-Steiner, Edelsbrunner, Harer, Mileyko (2010). *Found. Comput. Math.* 10:127. Lipschitz functions have L_p-stable persistence.
- Bubenik (2015). *J. Mach. Learn. Res.* 16:77. Statistical TDA using persistence landscapes.
- Bubenik, Vejdemo-Johansson (2018). *Topol. Methods Nonlinear Anal.* 51:151. Categorification of persistent homology (fast O(n log n) landscapes).
- Adams et al. (2017). *J. Mach. Learn. Res.* 18:1. Persistence images.
- Reininghaus, Huber, Bauer, Kwitt (2015). *CVPR*. PSSK — stable multi-scale kernel.
- Carrière, Cuturi, Oudot (2017). *ICML*. Sliced Wasserstein kernel for persistence diagrams.
- Le, Yamada (2018). *NeurIPS*. Persistence Fisher kernel.
- Kusano, Hiraoka, Fukumizu (2016). *ICML*. Persistence-weighted Gaussian kernel.
- Turner, Mileyko, Mukherjee, Harer (2014). *Discrete Comput. Geom.* 52:44. Frechet means for distributions of persistence diagrams.
- Mileyko, Mukherjee, Harer (2011). *Inverse Problems* 27:124007. Probability measures on the space of persistence diagrams.
- Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan, Singh (2014). *Ann. Stat.* 42:2301. Confidence sets for persistence diagrams.
- Robinson, Turner (2017). *J. Appl. Comput. Topol.* 1:241. Hypothesis testing for TDA.
- Chazal, Glisse, Labruere, Michel (2014). *J. Mach. Learn. Res.* 15:3603. Convergence rates for persistence diagram estimation.
- Chen, Genovese, Wasserman (2015). *Electron. J. Stat.* 9:1399. Statistical analysis of persistence intensity functions.
- Atienza, Gonzalez-Diaz, Soriano-Trigueros (2019). *Pattern Recognit.* 87:217. Stability of persistent entropy.
- Kerber, Morozov, Nigmetov (2017). *J. Exp. Algorithmics* 22:1.4. Hera (BSD-3 C++).
- scikit-tda persim 0.4 (Apache-2 Python). GUDHI 3.10 (Apache-2/MIT). giotto-tda 0.6 (AGPL).

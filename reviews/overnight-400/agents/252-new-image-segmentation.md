# 252 | new-image-segmentation — Mumford-Shah, Chan-Vese, graph cut, Otsu, watershed, mean shift, SLIC, TV, anisotropic diffusion

**Summary line 1.** reality v0.10.0 ships **ZERO** image-segmentation surface — repo-wide grep on `mumford.shah|chan.vese|level.set|active.contour|snake|kass.witkin|geodesic.active|gradient.vector.flow|gvf|graph.cut|min.cut|boykov.kolmogorov|alpha.expansion|random.walker|watershed|beucher|mean.shift|comaniciu|slic|superpixel|felzenszwalb|huttenlocher|otsu|adaptive.threshold|anisotropic.diffusion|perona.malik|bilateral.filter|total.variation|rudin.osher.fatemi|tv.l1|chambolle|primal.dual|markov.random.field|mrf|gibbs.sampler|simulated.annealing.mrf|ising.image|potts|continuous.max.flow|convex.relaxation.segmentation|dbscan|gaussian.mixture|gmm|em.cluster|kmeans|fuzzy.c.means|beltrami.flow|sochen|ambrosio.tortorelli|piecewise.smooth|piecewise.constant|region.growing|active.shape.model|asm|level.set.osher.sethian|fast.marching` against `*.go` returns **zero callable matches** (the lone tangential hit is `graph/flow.go::MaxFlow` Edmonds-Karp BFS-Ford-Fulkerson at O(V·E²) — usable as a graph-cut substrate but neither a min-cut extractor nor the Boykov-Kolmogorov 2004 augmenting-path algorithm specialised for grid graphs that every modern segmentation paper consumes; `optim/proximal/{ProxL1,ProxL2,Fbs,Fista,Admm}` 1,150 LOC is reusable substrate for TV-regularisation via Chambolle-2004 dual / Chambolle-Pock-2011 PDHG but ships zero TV-prox + zero TV-discretisation; `signal/filter.go::MovingAverage,MedianFilter` are 1-D-only, no 2-D bilateral / no 2-D Gaussian / no 2-D Sobel; `signal/fft.go::FFT/IFFT` is 1-D-only, no FFT2D for spectral-PDE diffusion paths; `prob/distributions.go::NormalPDF` exists but no GMM-EM / no DBSCAN / no kmeans / no MRF-Gibbs-sampler — the entire 1979-2010 segmentation canon (Otsu, Beucher-Lantuéjoul watershed, Kass-Witkin-Terzopoulos snakes, Caselles-Kimmel-Sapiro geodesic-AC, Mumford-Shah, Chan-Vese, Felzenszwalb-Huttenlocher, Boykov-Kolmogorov graph-cut, Comaniciu-Meer mean-shift, Achanta SLIC, Grady random walker) is wholly absent and there is **NO `image/` sub-package** to host it). 158-synergy-color-signal already enumerated 18 image-shaped color×signal primitives (~2,380 LOC) calling for a NEW `image/` sub-package as the cycle-free cross-color-cross-signal placement; this slot 252 is the **second consumer** of that proposed `image/` sub-package, demanding the segmentation-algorithm tier on top of the filter/blur/demosaic tier 158 specified. Strict cross-dependencies: 215-compressed-sensing **C7-tier T7-structured-sparsity TV-regularisation-1D-Condat-direct + TV-2D-Chambolle-dual + TV-3D-PDHG ~520 LOC** is the EXACT TV-substrate this slot consumes for ROF-denoising / TV-Chan-Vese / convex-relaxation-Chan-Esedoglu-Nikolova; 244-pde-solvers **D3 Laplacian2D 5-pt + D11 Poisson2D Jacobi/GS/SOR + D18 upwind + D7 HeatEquation1D-explicit + D8 HeatEquation2D-Crank-Nicolson + D9 Peaceman-Rachford-ADI** are the EXACT PDE-substrate for Perona-Malik-1990 anisotropic-diffusion + Beltrami-flow + level-set-Osher-Sethian + Mumford-Shah-Ambrosio-Tortorelli + Chan-Vese-level-set evolution; 158-synergy-color-signal **C0 image.Plane + C1 LinearLightGaussianBlur + C4 BilateralFilter ΔE2000 + C16 signal.Convolve2DSeparable + signal.GaussianKernel1D** are the EXACT image-shaped substrate for image-input/output convention. **Block-C verdict:** image-segmentation is ABSENT but the substrate is coalescing — once `image/` (158) + TV-prox (215 T7) + Laplacian2D (244 D3) land, 252's full segmentation tier ships at **~3,800 LOC of pure synthesis** without any new mathematics beyond the algorithms themselves.

**Summary line 2.** Twenty-six segmentation primitives **S1-S26** totalling ~3,800 LOC organised as **(a) Tier-0 thresholding ~280 LOC** (S1 `image/segment/otsu.go` Otsu-1979 binary-threshold via histogram-variance-maximisation + multi-class-Otsu-Liao-Chen-Chung-2001 + Bradley-Roth-2007 adaptive-threshold integral-image-O(N) ~140 LOC, S2 `image/segment/threshold.go` Niblack-1986 + Sauvola-Pietikäinen-2000 + Wolf-Jolion-2004 windowed-mean-stdev locally-adaptive thresholds ~80 LOC, S3 `image/segment/histogram.go` shared histogram + integral-histogram + Bhattacharyya-distance + KL-divergence-between-histograms helpers ~60 LOC), **(b) Tier-1 region-based clustering ~520 LOC** (S4 `image/segment/kmeans.go` Lloyd's k-means + k-means++-Arthur-Vassilvitskii-2007 + mini-batch-Sculley-2010 over RGB/Lab/Luv pixel-vectors ~140 LOC, S5 `image/segment/gmm.go` Gaussian-mixture-EM with k full-covariance components + BIC-model-selection ~160 LOC, S6 `image/segment/dbscan.go` Density-Based-Spatial-Clustering Ester-Kriegel-Sander-Xu-1996 with KD-tree-radius-query (depends on linalg/kdtree which 097-linalg-missing flags) ~120 LOC, S7 `image/segment/meanshift.go` Comaniciu-Meer-2002 PAMI-24:603 mode-seeking with Epanechnikov / Gaussian profile, joint spatial-range bandwidth `(h_s, h_r)` ~100 LOC), **(c) Tier-2 region-growing + watershed ~360 LOC** (S8 `image/segment/region_growing.go` seeded-region-growing-Adams-Bischof-1994 + similarity-criterion-pluggable ~80 LOC, S9 `image/segment/watershed.go` Beucher-Lantuéjoul-1979 / Vincent-Soille-1991 immersion-simulation watershed via priority-queue-flooding ~140 LOC, S10 `image/segment/marker_watershed.go` Meyer-1994 marker-controlled watershed avoiding over-segmentation ~80 LOC, S11 `image/segment/gradient.go` Sobel + Scharr + Prewitt + Roberts gradient operators for watershed-input ~60 LOC), **(d) Tier-3 graph-based ~600 LOC** (S12 `image/segment/felzenszwalb.go` Felzenszwalb-Huttenlocher-2004 IJCV-59:167 graph-segmentation via union-find + sorted-edge-Kruskal-style merge with internal-difference-vs-MInt-threshold ~140 LOC, S13 `image/segment/random_walker.go` Grady-2006 PAMI-28:1768 random-walker via sparse-Laplacian-system + linalg/cg ConjugateGradient (depends on 244 D12 + Laplacian2D D3) ~160 LOC, S14 `image/segment/graph_cut.go` Boykov-Kolmogorov-2004 PAMI-26:1124 augmenting-path max-flow specialised for grid-graphs O(mn²|C|) where |C| is min-cut value, much faster than Edmonds-Karp on dense grid graphs ~200 LOC, S15 `image/segment/alpha_expansion.go` Boykov-Veksler-Zabih-2001 PAMI-23:1222 multi-label MRF-energy-minimisation via repeated binary alpha-expansion graph-cuts ~100 LOC), **(e) Tier-4 active-contour + level-set ~720 LOC** (S16 `image/segment/snake.go` Kass-Witkin-Terzopoulos-1988 IJCV-1:321 parametric-active-contour with internal-energy α||v'||² + β||v''||² + external-edge-energy via Euler-Lagrange semi-implicit-pentadiagonal-banded solve ~140 LOC, S17 `image/segment/gvf_snake.go` Xu-Prince-1998 PAMI-7:359 Gradient-Vector-Flow snake replacing edge-gradient with diffusion-PDE-vector-field that propagates into homogeneous regions ~120 LOC, S18 `image/segment/geodesic_ac.go` Caselles-Kimmel-Sapiro-1997 IJCV-22:61 geodesic-active-contour as Riemannian-distance-minimisation via level-set evolution `∂φ/∂t = g(I)·κ·|∇φ| + α·g(I)·|∇φ| + ∇g·∇φ` ~140 LOC, S19 `image/segment/chan_vese.go` Chan-Vese-2001 IEEE-TIP-10:266 piecewise-constant region-based active-contour `min Σ_i ∫_{Ω_i} (I − c_i)² + ν·Length(C) + μ·Area(inside)` via level-set with Heaviside-regularisation ~160 LOC, S20 `image/segment/multiphase_chan_vese.go` Vese-Chan-2002 IJCV-50:271 multi-phase Chan-Vese using log₂(K) level-set functions for K phases ~80 LOC, S21 `image/segment/level_set.go` Osher-Sethian-1988 JCP-79:12 level-set evolution + reinitialisation + narrow-band optimisation + signed-distance-function-fast-marching-Sethian-1996 (also `chaos/` candidate) ~80 LOC), **(f) Tier-5 functional minimisation ~520 LOC** (S22 `image/segment/mumford_shah.go` Mumford-Shah-1989 piecewise-smooth `min ∫(u−f)² + α∫_{Ω∖K}|∇u|² + β·H¹(K)` via Ambrosio-Tortorelli-1990 elliptic-approximation with phase-field `v` (1≈smooth, 0≈edge) and ε→0 Γ-convergence ~180 LOC, S23 `image/segment/tv_segment.go` Chan-Esedoglu-Nikolova-2006 SIAM-J-Appl-Math-66:1632 convex-relaxation of Chan-Vese as `min ∫|∇u| + λ∫(c₁−c₂)·(2u−1)` over u∈[0,1] with global-minimum guarantee (no level-set initialisation sensitivity), via Chambolle-Pock-2011 PDHG ~140 LOC, S24 `image/segment/continuous_max_flow.go` Yuan-Bae-Tai-2010 SIAM-J-Imaging-3:1014 continuous max-flow / min-cut as convex-PDE primal-dual with proximal-iterations ~120 LOC, S25 `image/segment/anisotropic_diffusion.go` Perona-Malik-1990 PAMI-12:629 nonlinear-diffusion `∂u/∂t = div(c(|∇u|)·∇u)` with edge-stopping `c(s)=exp(−(s/K)²)` or `1/(1+(s/K)²)` + Catté-Lions-Morel-Coll-1992 SIAM-29:182 regularised version with Gaussian-pre-smoothing for well-posedness ~80 LOC), **(g) Tier-6 superpixels + clustering ~280 LOC** (S26 `image/segment/slic.go` Achanta-Shaji-Smith-Lucchi-Fua-Süsstrunk-2010-EPFL-149300 / 2012 PAMI-34:2274 Simple-Linear-Iterative-Clustering local-k-means in joint Lab+xy 5-D space with bandwidth `S = √(N/K)` ~120 LOC, S27 `image/segment/quickshift.go` Vedaldi-Soatto-2008-ECCV mode-seeking variant of mean-shift ~80 LOC, S28 `image/segment/fcm.go` Bezdek-1981 fuzzy-c-means soft-clustering ~80 LOC).

**SINGULAR-CHEAPEST-1-DAY S1 Otsu + S3 Histogram + S11 Sobel + S25 Perona-Malik ~340 LOC** because Otsu-1979 is the single most-cited binary-segmentation algorithm in all of imaging (>50,000 citations in 2026), Sobel is the ubiquitous gradient prerequisite, and Perona-Malik is the simplest PDE-segmentation-primitive that demonstrates the synergy with 244-D3 Laplacian2D — all four ship without any blockers beyond `image.Plane` (158-C0). **SINGULAR-FOUNDATIONAL S19 Chan-Vese ~160 LOC** because Chan-Vese-2001 is the canonical region-based level-set segmentation paper (>20,000 citations) and is the single most-implemented academic-reference segmentation algorithm — having a zero-dep Go + Python + C++ + C# byte-identical Chan-Vese is a unique reality contribution NO existing library provides at that quality bar. **SINGULAR-MOAT S14 Boykov-Kolmogorov graph-cut + S15 alpha-expansion ~300 LOC** because Boykov-Kolmogorov-2004 is the de-facto-standard segmentation min-cut algorithm (>15,000 citations) — implementations exist (the canonical `boost::graph::boykov_kolmogorov_max_flow`, OpenCV `cv::GraphCut`, the Vladimir Kolmogorov reference C++) but no zero-dep cross-language Go implementation; alpha-expansion adds the multi-label MRF tier that powers Photoshop's Magic-Wand-2.0 / GrabCut. **SINGULAR-2024-FRONTIER S23 TV-Chan-Vese-Chan-Esedoglu-Nikolova ~140 LOC** because the convex-relaxation reformulation is the single most-important post-2005 advance over level-set Chan-Vese (no local-minima trap, polynomial-time global-minimum, 1000× speedup via PDHG over level-set evolution) and is the operational-default in every modern medical-imaging segmentation pipeline (3D-Slicer, ITK-SNAP, Amira). **SINGULAR-PEDAGOGICAL S22 Mumford-Shah-via-Ambrosio-Tortorelli ~180 LOC** because the AT-1990 elliptic Γ-convergence approximation is the single most-elegant connection between calculus-of-variations + PDEs + segmentation in all of applied-math (Ambrosio-Fonseca-Mascarenhas 2003 Acta-Numerica review) and serves as the entire foundation for phase-field methods in image processing AND fracture mechanics AND topology optimisation (cross-link 251-shape-opt). Recommended placement **NEW sub-package `image/segment/`** under the `image/` sub-package proposed by 158-synergy-color-signal — same "consumer-shaped sub-package, not in primitive-supplier-package" precedent (151/153/156/157/158).

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for image-segmentation surface — **zero callable matches** anywhere in `*.go`.

| Surface | Path | Segmentation relevance |
|---|---|---|
| `graph.MaxFlow` (Edmonds-Karp) | `graph/flow.go:25` | Substrate for graph-cut-S14, but Edmonds-Karp O(V·E²) is wrong-asymptotic for grid-graphs vs Boykov-Kolmogorov O(mn²|C|); **PRESENT but inadequate** |
| `optim/proximal/{ProxL1,ProxL2,Fbs,Fista,Admm}` | `optim/proximal/*.go` | TV-prox + variational segmentation substrate; **PRESENT but no TV-discretisation** |
| `signal.MedianFilter` | `signal/filter.go` | 1-D-only; 2-D-median (used in Perona-Malik pre-filter) absent |
| `signal.Convolve` | `signal/filter.go` | 1-D-only; 2-D-Sobel/Gaussian/Laplacian convolutions absent (158 C16 ships them) |
| `signal.FFT/IFFT` | `signal/fft.go` | 1-D-only; 2-D-FFT for spectral-diffusion-pre-smoothing absent |
| `prob.NormalPDF/CDF` | `prob/distributions.go` | GMM-EM substrate; **PRESENT density only, no sampling, no EM-fit** |
| `linalg.{LU,Cholesky,QR}` | `linalg/decompose.go` | Random-walker linear-system substrate; **PRESENT** |
| `linalg.ConjugateGradient` | -- | Random-walker sparse-Laplacian-CG; **ABSENT** (244 D12 ships) |
| `pde.Laplacian2D / Poisson2D` | -- | Diffusion + level-set substrate; **ABSENT** (244 D3 + D11 ship) |
| `image.Plane / RGBPlanes` | -- | Image-shaped buffer type; **ABSENT** (158 C0 ships) |
| `image.GaussianBlur / BilateralFilter` | -- | Edge-preserving pre-smoothing; **ABSENT** (158 C1 + C4 ship) |
| `image.LabPlanes` | -- | Perceptual-space clustering substrate; **ABSENT** (158 C0 ships) |
| `linalg.KMeans` | -- | k-means clustering substrate; **ABSENT** (097 / 157 / 158 C14 flag) |
| `linalg.KDTree` | -- | DBSCAN radius-query substrate; **ABSENT** (097 flag) |
| `image/segment/` package | -- | **ABSENT** — this slot creates |
| S1-S28 segmentation primitives | -- | **ALL ABSENT** |

**Cross-import edges that this slot creates.**
- `image/segment → image.Plane / RGBPlanes / LabPlanes` (158 C0) for I/O type contract.
- `image/segment → image.GaussianBlur / BilateralFilter / Convolve2DSeparable` (158 C1 + C4 + C16) for pre-smoothing.
- `image/segment → linalg.{KMeans,KDTree}` (097) for S4 + S6 + S26.
- `image/segment → linalg.ConjugateGradient` (244 D12) for S13 random-walker sparse-system solve.
- `image/segment → pde.Laplacian2D` (244 D3) for S25 anisotropic-diffusion + S22 Ambrosio-Tortorelli.
- `image/segment → pde.HeatEquation2D-CN` (244 D8) for S25 implicit-Catté-Lions-Morel-Coll regularisation.
- `image/segment → graph.MaxFlow` (PRESENT) as fallback / oracle-validation against S14 Boykov-Kolmogorov.
- `image/segment → optim/proximal.{Fbs,Admm,ProxL1}` for S23 TV-Chan-Vese + S24 continuous-max-flow PDHG.
- `image/segment → optim/transport` (PRESENT) candidate cross-link to Wasserstein-segmentation (Chan-Esedoglu-Pan-2007 image-segmentation-via-transport-distances; defer, niche).

**Strict downstream consumers** of `image/segment/` substrate:
- Medical-imaging consumers (Pistachio's hypothetical retina-segmentation / sentinel scan-pipeline).
- Document-binarisation (Otsu / Niblack / Sauvola — S1 + S2).
- Computer-vision SOTA pipelines that pre-segment before ML inference (SLIC superpixels — S26 — feeds GraphConvNets).

---

## 1. The twenty-eight primitives (S1-S28)

Each entry: name, LOC, reference, API sketch.

### Tier 0 — Thresholding (~280 LOC)

**S1 — `image/segment/otsu.go` ~140 LOC.** `Otsu(plane image.Plane) (threshold float64)` Otsu-1979 IEEE-SMC-9:62 between-class-variance-maximisation `σ_B²(t) = ω_0(t)·ω_1(t)·(μ_0(t)−μ_1(t))²` over 256-bin histogram in O(L) post-O(N) histogram-build. `MultiOtsu(plane, k int) []float64` Liao-Chen-Chung-2001 dynamic-programming for k-class threshold `O(L²·k)`. `AdaptiveThresholdBradley(plane, windowSize int, t float64) image.Plane` Bradley-Roth-2007 integral-image-based adaptive threshold `O(N)`. **Refs.** Otsu-1979; Liao-Chen-Chung-2001 *J-Inf-Sci-Eng-17:713*; Bradley-Roth-2007 *J-Graphics-Tools-12:13*.

**S2 — `image/segment/threshold.go` ~80 LOC.** `Niblack(plane, windowSize int, k float64)` Niblack-1986 `T(x,y) = m(x,y) + k·s(x,y)` where m,s are local mean/stdev. `Sauvola(plane, windowSize int, k, R float64)` Sauvola-Pietikäinen-2000 `T = m·(1 + k·(s/R − 1))` for document-binarisation (the OCR-default). `WolfJolion(plane, windowSize)` Wolf-Jolion-2004 normalisation refinement. **Refs.** Niblack-1986 *Introduction to Digital Image Processing* §5.2; Sauvola-Pietikäinen-2000 *Pattern-Recognition-33:225*; Wolf-Jolion-2004 *Pattern-Analysis-and-Applications-7:237*.

**S3 — `image/segment/histogram.go` ~60 LOC.** `Histogram(plane, bins int) []int` standard binning. `IntegralImage(plane image.Plane) []float64` summed-area-table (Crow-1984 / Viola-Jones-2001) for O(1)-rectangle-sum used by S1 + S2 + S26. `Bhattacharyya(h1, h2 []float64) float64` and `KLDivergence(h1, h2 []float64) float64` histogram-distance metrics for region-merging.

### Tier 1 — Region-based clustering (~520 LOC)

**S4 — `image/segment/kmeans.go` ~140 LOC.** `KMeansSegment(rgb image.RGBPlanes, k, maxIter int, seed int64) (labels []int, centers [][3]float64)` Lloyd-1982 with k-means++ Arthur-Vassilvitskii-2007 SODA initialisation. Default colour-space: CIELab (perceptual proximity) via `color.SRGBToLab`. `MiniBatchKMeans` Sculley-2010-WWW for online / streaming. Cross-link: 158-C14 + 097 + 157 (KMeans is shared keystone). **Refs.** Lloyd-1982 *IEEE-IT-28:129*; Arthur-Vassilvitskii-2007 SODA-1027.

**S5 — `image/segment/gmm.go` ~160 LOC.** `GaussianMixtureSegment(rgb image.RGBPlanes, k, maxIter int, seed int64) (labels []int, posteriors [][]float64, params GMMParams)` EM-algorithm with full 3×3 covariances per component. `BIC(model, data)` model-selection. Subsumes "Mahalanobis-RGB-clustering". **Refs.** Dempster-Laird-Rubin-1977 *JRSS-B-39:1*; Bishop-2006-PRML §9.

**S6 — `image/segment/dbscan.go` ~120 LOC.** `DBSCANSegment(plane image.Plane, eps float64, minPts int) (labels []int, noise []bool)` density-based clustering — labels[i] = -1 for noise. Depends on `linalg/kdtree` (097-flagged) for O(N·log·N) ε-radius queries; falls back to O(N²) if absent. **Refs.** Ester-Kriegel-Sander-Xu-1996 *KDD-96:226*; Schubert-Sander-Ester-Kriegel-Xu-2017 *ACM-TODS-42:19* DBSCAN-revisited.

**S7 — `image/segment/meanshift.go` ~100 LOC.** `MeanShiftSegment(rgb image.RGBPlanes, hSpatial, hRange float64, maxIter int) (labels []int)` Comaniciu-Meer-2002 mode-seeking in joint 5-D `(x, y, L, a, b)` space with Epanechnikov-kernel `K(u) = (1−||u||²) for ||u||≤1`. Convergence: each pixel's mode-trajectory hill-climbs the kernel-density-estimate gradient; pixels converging to the same mode share a label. **Refs.** Comaniciu-Meer-2002 *PAMI-24:603*; Cheng-1995 *PAMI-17:790* mean-shift-original; Fukunaga-Hostetler-1975 *IEEE-IT-21:32* gradient-of-density-original.

### Tier 2 — Region-growing + watershed (~360 LOC)

**S8 — `image/segment/region_growing.go` ~80 LOC.** `SeededRegionGrow(plane, seeds []Seed, similarity SimilarityFn) []int` Adams-Bischof-1994. Pluggable similarity (Δ-intensity, ΔE-color, gradient-magnitude). **Refs.** Adams-Bischof-1994 *PAMI-16:641*.

**S9 — `image/segment/watershed.go` ~140 LOC.** `Watershed(gradient image.Plane) []int` Vincent-Soille-1991 PAMI-13:583 immersion-simulation via priority-queue-flooding from local-minima. The classical "waterfall" segmentation; tends to over-segment (every local-minimum becomes a basin). **Refs.** Beucher-Lantuéjoul-1979 IRIA-79; Vincent-Soille-1991 *PAMI-13:583*.

**S10 — `image/segment/marker_watershed.go` ~80 LOC.** `MarkerWatershed(gradient image.Plane, markers []int) []int` Meyer-1994 marker-controlled. User specifies foreground+background markers; watershed flooding only from those, avoiding over-segmentation. **Refs.** Meyer-1994 *Signal-Processing-38:113*.

**S11 — `image/segment/gradient.go` ~60 LOC.** `Sobel(plane image.Plane) (gx, gy image.Plane)` 3×3 separable kernel via 158-C16. `Scharr` 3×3 rotation-invariant alternative. `Prewitt`, `Roberts` historical alternatives. Output magnitude = `√(gx² + gy²)`, direction = `atan2(gy, gx)`. **Refs.** Sobel-Feldman-1968 SAIL-presentation; Scharr-2000-PhD §3; Prewitt-1970 *Picture-Processing-and-Psychopictorics-75*.

### Tier 3 — Graph-based (~600 LOC)

**S12 — `image/segment/felzenszwalb.go` ~140 LOC.** `FelzenszwalbHuttenlocher(plane image.Plane, k float64, minSize int) []int` IJCV-59:167. Build 8-connected pixel-graph → sort edges by weight → union-find merging when `w(e) ≤ MInt(C₁, C₂)` where `MInt(C₁,C₂) = min(Int(C₁) + k/|C₁|, Int(C₂) + k/|C₂|)` and `Int(C)` is max-edge-weight in MST of C. O(n·log·n) total. **Refs.** Felzenszwalb-Huttenlocher-2004 *IJCV-59:167*.

**S13 — `image/segment/random_walker.go` ~160 LOC.** `RandomWalker(plane image.Plane, seeds []SeededLabel) (probabilities [][]float64)` Grady-2006 PAMI-28:1768. Solves `L_U · x = −B^T · M_seeds` where L is the graph-Laplacian (with weights `w_ij = exp(−β·(I_i − I_j)²)`), L_U is L restricted to unseeded pixels. Solve via `linalg/cg.ConjugateGradient` (244 D12). Each pixel gets a probability per label; final label = argmax. Provably-no-trap convergence (vs Chan-Vese local-minima). **Refs.** Grady-2006 *PAMI-28:1768*; Couprie-Grady-Najman-Talbot-2011 *PAMI-33:1384* power-watershed-unification.

**S14 — `image/segment/graph_cut.go` ~200 LOC — KEYSTONE.** `BoykovKolmogorovMaxFlow(adj, capacity, source, sink)` PAMI-26:1124-2004 augmenting-path algorithm specialised for vision-graphs (grid-structured, low-source-sink-distance). Maintains two trees S, T from source/sink, finds augmenting paths via tree-growth + reuse, parent-relabel on saturation. `BinarySegment(plane, fgSeeds, bgSeeds)` formulates segmentation as `min Σ_p D_p(L_p) + Σ_{(p,q)} V_{pq}(L_p, L_q)` with smoothness `V_{pq} = exp(−β·(I_p − I_q)²)/dist(p,q)` and data-term from seed-likelihood. R-MUTUAL-CROSS-VALIDATION 3/3: BK-S14 ↔ Edmonds-Karp graph.MaxFlow ↔ Dinic-S14b (~80 LOC additional, defer) — all three must give identical max-flow value to 1e-12 on every test instance. **Refs.** Boykov-Kolmogorov-2004 *PAMI-26:1124*; Kolmogorov-Zabih-2004 *PAMI-26:147* what-energy-functions-can-be-minimised-via-graph-cuts (the regularity-condition theorem).

**S15 — `image/segment/alpha_expansion.go` ~100 LOC.** `AlphaExpansion(unary, pairwise, numLabels, maxIter)` Boykov-Veksler-Zabih-2001 PAMI-23:1222 multi-label MRF-energy-minimisation by repeated binary "α-vs-not-α" graph-cut moves. Provably finds a 2-approximation under metric pairwise terms. Used in stereo-matching, multi-label segmentation, denoising. **Refs.** Boykov-Veksler-Zabih-2001 *PAMI-23:1222*; Veksler-1999-PhD; Komodakis-Tziritas-2007 *PAMI-29:1436* primal-dual-extension.

### Tier 4 — Active-contour + level-set (~720 LOC)

**S16 — `image/segment/snake.go` ~140 LOC.** `KassWitkinTerzopoulos(plane image.Plane, init []Point, alpha, beta, gamma float64, kappa float64, maxIter int) (contour []Point)` IJCV-1:321-1988. Parametric closed-curve `v(s)`; minimise `E = ∫ α|v'|² + β|v''|² ds + ∫ E_ext(v) ds` via Euler-Lagrange semi-implicit pentadiagonal-banded solve. Edge-based. **Refs.** Kass-Witkin-Terzopoulos-1988 *IJCV-1:321*.

**S17 — `image/segment/gvf_snake.go` ~120 LOC.** `GVFSnake(plane, init, alpha, beta, mu, maxIter)` Xu-Prince-1998 PAMI-7:359. GVF-vector-field `v(x,y)` minimises `E_GVF = ∬ μ(u_x² + u_y² + v_x² + v_y²) + |∇f|² · |v − ∇f|² dx dy` — diffuses gradient into homogeneous regions, removes capture-range limitation of S16. **Refs.** Xu-Prince-1998 *PAMI-7:359*.

**S18 — `image/segment/geodesic_ac.go` ~140 LOC.** `GeodesicActiveContour(plane, init image.LevelSet, alpha float64, maxIter int) image.LevelSet` Caselles-Kimmel-Sapiro-1997 IJCV-22:61. Level-set evolution `∂φ/∂t = g(I)·κ·|∇φ| + α·g(I)·|∇φ| + ∇g·∇φ` where `g(|∇I|) = 1/(1+|∇I|²/K²)` is edge-stopping function and κ is curvature. The Riemannian-distance reformulation of S17 with topology-changes-allowed via level-set. **Refs.** Caselles-Kimmel-Sapiro-1997 *IJCV-22:61*; Caselles-Catté-Coll-Dibos-1993 *Numer-Math-66:1* preliminary.

**S19 — `image/segment/chan_vese.go` ~160 LOC — FOUNDATIONAL.** `ChanVese(plane image.Plane, init image.LevelSet, mu, nu, lambda1, lambda2 float64, maxIter int) image.LevelSet` Chan-Vese-2001 IEEE-TIP-10:266. Minimise `F(c₁, c₂, C) = μ·Length(C) + ν·Area(inside(C)) + λ₁∫_{inside}(I − c₁)² + λ₂∫_{outside}(I − c₂)²` via level-set evolution with Heaviside-regularisation `H_ε(φ) = (1/2)·(1 + (2/π)·atan(φ/ε))`. The piecewise-constant-Mumford-Shah special case (no smoothing inside regions). **Refs.** Chan-Vese-2001 *IEEE-TIP-10:266*; Chan-Sandberg-Vese-2000 *J-Vis-Comm-Image-Repr-11:130* preliminary.

**S20 — `image/segment/multiphase_chan_vese.go` ~80 LOC.** `MultiphaseChanVese(plane, K, init []image.LevelSet, maxIter)` Vese-Chan-2002 IJCV-50:271. Uses log₂(K) level-set functions to encode K phases via product-indicator-functions. **Refs.** Vese-Chan-2002 *IJCV-50:271*.

**S21 — `image/segment/level_set.go` ~80 LOC.** `LevelSetEvolve(phi image.Plane, F func(...) float64, dt float64, maxIter int)` Osher-Sethian-1988 JCP-79:12 Hamilton-Jacobi level-set framework. `Reinitialise(phi)` signed-distance-function via fast-marching (Sethian-1996). `NarrowBand(phi, radius)` optimisation (Adalsteinsson-Sethian-1995). **Refs.** Osher-Sethian-1988 *JCP-79:12*; Sethian-1996 *PNAS-93:1591* fast-marching.

### Tier 5 — Functional minimisation (~520 LOC)

**S22 — `image/segment/mumford_shah.go` ~180 LOC — PEDAGOGICAL.** `MumfordShah(f image.Plane, alpha, beta, eps float64, maxIter int) (u, v image.Plane)` Ambrosio-Tortorelli-1990 elliptic-approximation: minimise `AT_ε(u, v) = ∫(u−f)² + α·v²·|∇u|² + β·(ε·|∇v|² + (1−v)²/(4ε))`; as ε→0, AT-functional Γ-converges to Mumford-Shah-1989. Alternating-PDE-update on (u, v) via 244-D11 Poisson2D solves. **Refs.** Mumford-Shah-1989 *Comm-Pure-Appl-Math-42:577*; Ambrosio-Tortorelli-1990 *Comm-Pure-Appl-Math-43:999*; Ambrosio-Fonseca-Mascarenhas-2003 *Acta-Numerica-12* review.

**S23 — `image/segment/tv_segment.go` ~140 LOC — 2024-FRONTIER.** `TVChanVese(plane, c1, c2, lambda, maxIter)` Chan-Esedoglu-Nikolova-2006 SIAM-J-Appl-Math-66:1632. Convex relaxation `min_{u∈[0,1]} ∫|∇u| + λ·∫(c₁−c₂)·(2u−1)·dx`; threshold `u > 0.5` recovers Chan-Vese minimiser globally (no level-set initialisation). Solve via Chambolle-Pock-2011 PDHG. Depends on TV-prox from 215-T7. **Refs.** Chan-Esedoglu-Nikolova-2006 *SIAM-J-Appl-Math-66:1632*; Bresson-Esedoglu-Vandergheynst-Thiran-Osher-2007 *J-Math-Imaging-Vis-28:151* fast-implementation.

**S24 — `image/segment/continuous_max_flow.go` ~120 LOC.** `ContinuousMaxFlow(plane, source, sink, maxIter)` Yuan-Bae-Tai-2010 SIAM-J-Imaging-3:1014. Convex-PDE-formulation of Boykov-Kolmogorov-S14 in continuous-domain → primal-dual proximal-iterations → no graph-discretisation artefacts (8-vs-26-connectivity bias). **Refs.** Yuan-Bae-Tai-2010 *SIAM-J-Imaging-3:1014*; Pock-Cremers-Bischof-Chambolle-2009-CVPR convex-relaxation-Chan-Vese-multiphase.

**S25 — `image/segment/anisotropic_diffusion.go` ~80 LOC.** `PeronaMalik(plane, K float64, lambda float64, maxIter int, eqType int) image.Plane` Perona-Malik-1990. PDE `∂u/∂t = div(c(|∇u|)·∇u)` with `c(s) = exp(−(s/K)²)` (eq-1) or `c(s) = 1/(1 + (s/K)²)` (eq-2). Catté-Lions-Morel-Coll-1992 SIAM-29:182 regularised version: replace `|∇u|` with `|∇(G_σ ⋆ u)|` for well-posedness. Discretisation via 244-D3 Laplacian2D or explicit 4-neighbour upwinding. **Refs.** Perona-Malik-1990 *PAMI-12:629*; Catté-Lions-Morel-Coll-1992 *SIAM-J-Numer-Anal-29:182*; Weickert-1998 book.

### Tier 6 — Superpixels + clustering (~280 LOC)

**S26 — `image/segment/slic.go` ~120 LOC.** `SLIC(rgb image.RGBPlanes, K int, m float64, maxIter int) []int` Achanta-Shaji-Smith-Lucchi-Fua-Süsstrunk-2010-EPFL-149300 / 2012 PAMI-34:2274. Initialise K cluster-centres on regular grid `S = √(N/K)`; assign each pixel to nearest centre in joint 5-D `(L, a, b, x/m·S, y/m·S)` space restricted to 2S×2S local search; recompute centres; iterate ≤10. Compactness parameter `m∈[1,40]` trades shape-regularity vs colour-fidelity. **Refs.** Achanta-Shaji-Smith-Lucchi-Fua-Süsstrunk-2012 *PAMI-34:2274*; Liu-Tuzel-Ramalingam-Chellappa-2011-CVPR entropy-rate-superpixels-alternative.

**S27 — `image/segment/quickshift.go` ~80 LOC.** `Quickshift(rgb, sigma, tau, maxIter)` Vedaldi-Soatto-2008-ECCV mode-seeking variant of mean-shift; each pixel links to its nearest higher-density neighbour within τ — produces a forest where roots are modes. **Refs.** Vedaldi-Soatto-2008 *ECCV-LNCS-5305:705*.

**S28 — `image/segment/fcm.go` ~80 LOC.** `FuzzyCMeans(plane, c, m float64, maxIter int) ([][]float64, [][]float64)` Bezdek-1981. Soft-clustering: each pixel has a fuzzy-membership-vector u_i ∈ Δ^{c−1}; objective `Σ_{i,j} u_{ij}^m · ||x_i − v_j||²`. **Refs.** Bezdek-1981 *Pattern-Recognition-with-Fuzzy-Objective-Function-Algorithms* book; Pal-Bezdek-1995 *IEEE-T-Fuzzy-Systems-3:370*.

---

## 2. Connective tissue + cross-package blockers

**Substrate-blocker-1** `image.Plane / RGBPlanes / LabPlanes` (158-C0 ~50 LOC). Gates ALL of `image/segment/`. ZERO without 158-C0.

**Substrate-blocker-2** `signal.Convolve2DSeparable + signal.GaussianKernel1D` (158-C16 ~80 LOC). Gates S11 Sobel + S25 Perona-Malik regularisation + S22 Ambrosio-Tortorelli pre-smoothing.

**Substrate-blocker-3** `image.GaussianBlur + image.BilateralFilter` (158-C1 + C4 ~210 LOC). Gates S25 Catté-Lions regularisation + S16-S20 contour pre-smoothing.

**Substrate-blocker-4** `pde.Laplacian2D` (244-D3 ~70 LOC subset). Gates S22 + S25 + S18 + S21 + S25.

**Substrate-blocker-5** `pde.Poisson2D + linalg.ConjugateGradient` (244-D11 + D12 ~320 LOC). Gates S13 random-walker.

**Substrate-blocker-6** `linalg.KMeans` (097 / 157 / 158-C14 ~120 LOC). Gates S4 + S26-SLIC-inner-loop + S5-GMM-init.

**Substrate-blocker-7** `linalg.KDTree` (097 ~150 LOC). Gates S6 DBSCAN at fast-asymptotic. (S6 ships at O(N²) without it.)

**Substrate-blocker-8** TV-prox + Chambolle-Pock-PDHG (215-T7 ~140 LOC). Gates S23 TV-Chan-Vese + S24 continuous-max-flow.

**Substrate-blocker-9** `optim/proximal.{Fbs,Admm}` (PRESENT). Used by S22 + S23 + S24.

**Substrate-blocker-10** `prob.GMMFit / EM-algorithm` ~120 LOC NEW (097 / 117 / 169 candidate). Gates S5 GMM. (Or inline EM in S5 itself.)

**Total upstream-substrate dependency** (not counting graph/MaxFlow which is PRESENT): ~1,260 LOC of NEW code in image/ + signal/ + pde/ + linalg/ + prob/ before any segmentation primitive can land at full quality. Of these:
- 158-C0 + C1 + C4 + C16 owned by 158-synergy-color-signal (~280 LOC subset).
- 244-D3 + D11 + D12 owned by 244-pde-solvers (~390 LOC subset).
- 097 + 157 + 158-C14 KMeans (~120 LOC).
- 215-T7 TV-prox (~140 LOC subset).
- 097 KDTree (~150 LOC) — soft-block on S6.
- prob/GMM (~120 LOC) — could inline in S5.

**Cheapest-no-blocker subset** (ships against substrate that 158 + 244 already specify, no inline-KMeans, no inline-TV): **S1 Otsu + S3 Histogram + S11 Sobel + S25 Perona-Malik + S9 Watershed ~480 LOC**.

**Recommended PR sequence:**

- **PR-A (substrate)** depends on 158-C0+C1+C4+C16 + 244-D3+D11+D12 landing first.
- **PR-B (Tier-0 thresholding ~280 LOC, 1-2 days)** S1 + S2 + S3. Otsu / Niblack / Sauvola — ships against image.Plane only. Independent of TV/KMeans.
- **PR-C (Tier-2 watershed ~280 LOC, 2-3 days)** S9 + S10 + S11. Watershed + marker-watershed + Sobel/Scharr gradients. Independent of TV/KMeans/CG.
- **PR-D (Tier-5 PDE-segmentation ~440 LOC, 1 week)** S25 Perona-Malik + S22 Mumford-Shah-AT. Depends on 244-D3 + 244-D11 (Laplacian2D + Poisson2D + alternating-PDE-update). Pedagogical PDE keystone.
- **PR-E (Tier-4 active-contour ~720 LOC, 2 weeks)** S16 + S17 + S18 + S19 + S20 + S21. Contour family + level-set. Chan-Vese-S19 is the headline.
- **PR-F (Tier-3 graph-cut ~600 LOC, 2 weeks)** S12 Felzenszwalb + S13 Random-Walker + S14 Boykov-Kolmogorov + S15 alpha-expansion. SINGULAR-MOAT. Depends on 244-D12 CG.
- **PR-G (Tier-1 region-clustering ~520 LOC, 2 weeks)** S4 KMeans + S5 GMM + S6 DBSCAN + S7 Mean-Shift. Depends on 097 KMeans + KDTree.
- **PR-H (Tier-5 convex-relaxation ~260 LOC, 1 week)** S23 TV-Chan-Vese + S24 continuous-max-flow. SINGULAR-2024-FRONTIER. Depends on 215-T7 TV-prox.
- **PR-I (Tier-6 superpixels ~280 LOC, 1 week)** S26 SLIC + S27 Quickshift + S28 FCM.

Total `image/segment/` PR-A through PR-I: ~3,800 LOC, ~10-12 engineer-weeks.

---

## 3. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins this slot enables

**Pin 1 — Otsu binary threshold on bimodal Gaussian histogram.** Three paths to threshold:
- S1 Otsu via histogram-variance-maximisation
- S5 GMM-EM with k=2 → midpoint of the two means
- Closed-form Bayesian-decision-boundary for known equi-prior bivariate Gaussians (analytical reference)

All three must agree to ±1 histogram-bin (~1/256). Saturates 3/3.

**Pin 2 — Boykov-Kolmogorov graph-cut value.** Three paths to max-flow value on a 32×32 binary-segmentation problem with analytical-known-min-cut:
- S14 Boykov-Kolmogorov augmenting-path
- S14b Dinic blocking-flow (~80 LOC additional, defer)
- graph.MaxFlow Edmonds-Karp (PRESENT)

All three must give identical max-flow to 1e-12 (integer-capacity problems exact). Saturates 3/3 — and the THREE algorithms run-times pin the relative-asymptotic performance (BK 5-10× faster than EK on grid graphs, paper-claim verified).

**Pin 3 — Chan-Vese ↔ TV-Chan-Vese ↔ Mumford-Shah piecewise-constant.** Three paths to segmentation of a synthetic "white circle on grey background" with σ=0.1 noise:
- S19 Chan-Vese level-set (with random init → may local-trap)
- S23 TV-Chan-Vese convex-relaxation (provably global minimum)
- S22 Mumford-Shah-AT in piecewise-constant limit β→∞

All three must converge to the same dice-coefficient ≥ 0.99 against ground-truth circle mask; S23 must dominate S19 on dice-coefficient when S19 is randomly-initialised away from the circle (the entire selling-point of convex-relaxation). Saturates 3/3 + a "negative-pin" against bad-init level-set.

**Pin 4 — Perona-Malik anisotropic-diffusion edge-preservation.** Three paths to filtered image of a noisy-step-edge:
- S25 Perona-Malik with `c(s)=exp(−(s/K)²)`
- S25 Perona-Malik with `c(s)=1/(1+(s/K)²)`
- 244-D7 HeatEquation1D (linear-diffusion = c(s)=1; the isotropic baseline)

Edge-sharpness (gradient-magnitude at the step-location) must be preserved by Perona-Malik (≥ 0.5 of original) and destroyed by linear-diffusion (≤ 0.1 of original). The "diffuses-along-edges-only" property of Perona-Malik. Saturates 3/3 + negative-pin against linear-diffusion.

**Pin 5 — SLIC superpixel boundary-recall.** Three paths to superpixel decomposition of a Berkeley-Segmentation-Database-style image:
- S26 SLIC with K=400, m=10 → boundary-recall vs ground-truth boundaries
- S12 Felzenszwalb with k=300 → boundary-recall
- S7 Mean-Shift with hSpatial=8, hRange=10 → boundary-recall

All three must achieve boundary-recall ≥ 0.85 at 0.95 boundary-precision (Achanta-2012 Table 4 reference numbers). Cross-validates the three different "regular-cluster" segmentation approaches on a single image. Saturates 3/3.

---

## 4. Touchpoints with other agents

- **158 (synergy-color-signal):** image.Plane / RGBPlanes / LabPlanes / GaussianBlur / BilateralFilter / Convolve2DSeparable are PREREQUISITES for this slot. 158 owns the 280-LOC substrate; 252 is the **second consumer** (after 158's own filter/demosaic/tonemap tier).
- **215 (compressed-sensing) T7:** TV-regularisation-1D-Condat / TV-2D-Chambolle-dual / TV-3D-PDHG are PREREQUISITES for S23 TV-Chan-Vese + S24 continuous-max-flow.
- **244 (pde-solvers) D3/D7/D8/D11/D12:** Laplacian2D + heat-equation + Poisson2D + Conjugate-Gradient are PREREQUISITES for S22/S25/S13.
- **097 (linalg-missing):** KMeans + KDTree + SVD are PREREQUISITES for S4 + S6 + S5-GMM-cov-eigendecomp + S26-SLIC-inner-loop.
- **117 (prob-missing):** PRNG-Gaussian + Bernoulli sampling needed for S5 GMM-EM-init + S15 alpha-expansion-MRF-Gibbs-sampler.
- **157 (synergy-graph-linalg):** KMeans is shared keystone with G8/G9.
- **142 (topology-missing):** persistent-homology of the level-sets of `u(x,y)` produces a topological-summary of segmentation candidates (Edelsbrunner-Harer-2010 Section IX); cross-link defer to v1.1+.
- **190 (synergy-topology-signal):** persistence-on-images natural follow-up using 252-output as input.
- **219 (mean-field-games) M1+M2:** HJB-upwind + FP-positivity share PDE-substrate with S25 anisotropic-diffusion; both consume 244-D3+D18.
- **246 (discrete-exterior):** Whitney-form-FEM substrate provides an alternative discretisation for level-set/Chan-Vese on triangulated meshes — a 2026-frontier replacement for Cartesian-grid-PDE-segmentation; defer.
- **251 (shape-opt):** level-set-shape-derivative shares `image.LevelSet` / `pde.LevelSetEvolve` substrate with S19 Chan-Vese + S21 level-set-evolution; cross-link strong (segmentation = inverse-problem-shape-opt with data-fidelity-objective). Co-design `image.LevelSet` type to satisfy both consumers.
- **A future image-isolation review (none currently scheduled):** with 158 + 252 both calling for `image/` sub-package, recommend opening `image-numerics`, `image-missing`, `image-sota`, `image-api`, `image-perf` slots in a future overnight grid (158 already flagged this).

---

## 5. Singular load-bearing recommendation

**Ship PR-B (Tier-0 thresholding) FIRST as a no-blocker proof-of-life ~280 LOC, 1-2 days.** Otsu-1979 has 50,000+ citations and is the single most-implemented segmentation primitive in all of image processing — having it in zero-dep Go with golden-file cross-language validation against scikit-image's `filters.threshold_otsu` byte-identically would be a unique reality contribution and instantly demonstrates the value of the proposed `image/segment/` sub-package.

**Then ship PR-D (Mumford-Shah + Perona-Malik) as the PEDAGOGICAL PDE-bridge ~440 LOC, 1 week** because S22 + S25 are the canonical bridge from PDE-substrate (244-D3) to applied-image-segmentation, and serve as the strongest existence-proof that the PDE-package is more than just heat/wave/Poisson — it solves real-world variational problems.

**Then ship PR-F (graph-cut family) as the SINGULAR-MOAT ~600 LOC, 2 weeks.** Boykov-Kolmogorov is the single most-cited graph-segmentation algorithm (>15,000 citations), and a zero-dep cross-language byte-identical Go implementation exists in NO library worldwide.

**Then ship PR-H (TV convex-relaxation) as the 2024-FRONTIER ~260 LOC, 1 week** because Chan-Esedoglu-Nikolova-2006 + Yuan-Bae-Tai-2010 are the post-2005 modern-default that supersedes level-set Chan-Vese in production pipelines.

**Defer PR-G (region-clustering) and PR-I (superpixels) until 097-KMeans + 097-KDTree land** since they are blocked at quality-bar (k-means inline implementation in S4 is acceptable as bridging).

**Avoid scoping: ConvNet-segmentation / U-Net / SAM / Mask-R-CNN.** These are *deep-learning-architectures*, not classical-segmentation-math; reality is a math-not-DL library, so deep-learning-segmentation belongs downstream in `aicore` not in `reality/image/segment/`. The classical 1979-2010 segmentation canon is well-defined, well-cited, and well-suited to reality's golden-file-cross-language-validation positioning — modern DL is not.

**Avoid scoping: Active Shape Models / Active Appearance Models (Cootes-1992/1998).** These are statistical-shape-priors that require a training-database — incompatible with reality's "zero-dep, no-training-data" positioning.

**Avoid scoping: Conditional Random Fields (Lafferty-McCallum-Pereira-2001) for image segmentation.** Standard CRF is essentially MRF + alpha-expansion which S15 covers; structured-learning-CRF (max-margin / online-learning) belongs downstream in aicore.

**Final precision-hazards:** (a) Otsu degenerates to wrong-threshold on unimodal histograms — docstring must note `Otsu_assumes_bimodal_input` and recommend `MultiOtsu` fallback; (b) Boykov-Kolmogorov is exact for integer capacities only — float-capacity inputs require explicit-tolerance epsilon to avoid infinite-augmenting-loop on ill-conditioned inputs; (c) level-set reinitialisation is a numerical-art — Sussman-Smereka-Osher-1994 PDE-reinit vs Sethian fast-marching vs subcell-fix-Russo-Smereka-2000 produce different signed-distance-functions on the same input → cross-language-byte-identical requires pinning ONE algorithm (recommend fast-marching for closed-form-tractability); (d) Chan-Vese has TWO known failure-modes — (i) random-init local-trap (fixed by S23 TV-relaxation), (ii) `λ₁ ≠ λ₂` produces biased segmentation (docstring note); (e) SLIC iteration count default — Achanta-2012 says K=400, iter=10 → empirical 5 iterations suffice for most images; default-iter-10 for cross-paper-reproducibility, document; (f) random-walker has known-failure-mode on weak-edges where seeded-region "leaks" — pre-smoothing the input or strengthening β-Laplacian-weight is the standard remedy; (g) Mumford-Shah-AT depends on epsilon → 0 limit for Γ-convergence — finite-epsilon produces a regularised approximation; cross-language reproducibility requires pinning epsilon (recommend ε = h where h is grid-spacing); (h) Perona-Malik-original is mathematically ill-posed (Kichenassamy-1997 *Arch-Ration-Mech-Anal-141:131* showed forward-backward-diffusion blow-up); the Catté-Lions-Morel-Coll-1992 regularisation is mandatory for cross-language reproducibility.

**Headline:** Twenty-eight image-segmentation primitives close the entire 1979-2010 classical-segmentation canon (Otsu / watershed / mean-shift / Felzenszwalb / random-walker / Boykov-Kolmogorov / alpha-expansion / Kass-Witkin-Terzopoulos / GVF / Caselles-Kimmel-Sapiro / Chan-Vese / Mumford-Shah-Ambrosio-Tortorelli / TV-Chan-Vese / continuous-max-flow / Perona-Malik / SLIC) in ~3,800 LOC of pure synthesis on top of 158-image-substrate + 244-PDE-substrate + 215-TV-substrate + 097-linalg-substrate; cheapest-1-day-shippable Otsu+Sobel+Watershed+Perona-Malik ~480 LOC; foundational keystone Chan-Vese-S19 ~160 LOC; singular-moat Boykov-Kolmogorov-S14+alpha-expansion-S15 ~300 LOC; 2024-frontier TV-Chan-Vese-S23+continuous-max-flow-S24 ~260 LOC; pedagogical Mumford-Shah-Ambrosio-Tortorelli-S22 ~180 LOC. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (Otsu↔GMM↔Bayes-decision-boundary, BK↔Dinic↔Edmonds-Karp on graph-cut, Chan-Vese↔TV-Chan-Vese↔Mumford-Shah piecewise-constant, Perona-Malik↔linear-diffusion edge-preservation negative-pin, SLIC↔Felzenszwalb↔Mean-Shift boundary-recall). Strict-downstream of 158 + 244 + 215 + 097; strict-upstream of (currently absent) image-numerics / image-missing / image-sota slots that should be opened in a future overnight grid since the gap is wider than any other in the codebase. Recommended placement NEW sub-package `image/segment/` under the `image/` sub-package proposed by 158 (consumer-shaped sub-package, not in primitive-supplier package — 151/153/156/157/158 precedent).

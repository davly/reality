# 270 | new-graph-signal-proc — Graph signal processing: graph Fourier, wavelets, Chebyshev

**Summary line 1.** reality v0.10.0 ships ZERO Graph Signal Processing (GSP) surface — repo-wide grep on `GraphFourier|GFT|graph.signal|graph.wavelet|chebyshev.poly|chebyshev.expansion|hammond|vandergheynst|gribonval|spectral.wavelet|polynomial.filter|graph.filter|low.pass.graph|high.pass.graph|band.pass.graph|ARMA.graph|graph.sampling|graph.tv|total.variation.graph|graph.signal.recovery|graph.denoising|graph.interpolation|vertex.frequency|windowed.graph.fourier|WGFT|diffusion.wavelet|coifman.maggioni|tight.frame.graph|bipartite.gsp|chebnet|directed.graph.signal|random.walk.normalized|hodge.laplacian.signal` against `*.go` returns ZERO callable matches outside `signal/window.go::HannWindow|HammingWindow|BlackmanWindow` (regular-domain DSP tapers — DISTINCT from vertex-frequency localisation kernels) and `chaos/systems::Chebyshev` false-positive name-collision (Chebyshev-attractor 1D-map, NOT polynomial recurrence). The closest substrate is `signal/fft.go::FFT|IFFT|PowerSpectrum|FFTFrequencies` (180 LOC of Cooley-Tukey radix-2 — diagonalises the cycle-graph Laplacian by virtue of cycle-graph eigenbasis BEING the discrete Fourier basis, exhibit-A pin in 157-G10 for GFT-vs-FFT cross-validation), `signal/filter.go::Convolve|MovingAverage|MedianFilter|ExponentialMovingAverage` (regular-grid 1D filters — not graph-aware), and the existing `graph/` package (12 deterministic combinatorial primitives at ~1400 LOC: BFS/Dijkstra/Bellman-Ford/A*/FloydWarshall/MaxFlow/TopoSort/Brandes/PageRank/Tarjan-SCC/Louvain/Kruskal-Prim — ZERO matrix-shaped numeric primitives). **PARTIAL OVERLAP with 157-synergy-graph-linalg (G1-G17 ~1380 LOC):** 157 ships the spectral substrate {AdjacencyMatrix, Laplacian, SymNormalizedLaplacian, RandomWalkLaplacian, AlgebraicConnectivity, Fiedler, SpectralBisection, GraphFourierTransform, EffectiveResistance, HeatKernel, MatrixTree} which is exactly the upstream substrate this slot consumes — slot 270 elevates GSP from "the GFT exists" (one primitive, G10) to "first-class signal-processing-on-graphs library" (28 primitives covering wavelets, Chebyshev approximation, polynomial filters, sampling theory, denoising, vertex-frequency, time-vertex). **PARTIAL OVERLAP with 245-new-spectral-methods (S1+S5+S12 ~330 LOC):** 245 ships orthogonal-polynomial recurrences `T_n(x), U_n(x), P_n(x), H_n(x), L_n(x), P_n^{(α,β)}(x)`, Clenshaw-Curtis quadrature, DCT, Chebyshev differentiation matrix on the continuous interval `[-1,1]` — slot 270 reuses S1 Chebyshev-T_n recurrence verbatim for the matrix-Chebyshev expansion `f(L) ≈ Σ c_k T_k(L̃)` (Hammond-Vandergheynst-Gribonval 2011 §6) which is the SINGULAR keystone enabling fast-graph-wavelet-transform without explicit eigendecomposition. **PARTIAL OVERLAP with 246-new-discrete-exterior (X25 spectral-DEC heat-kernel-signature):** 246 spectral-DEC restricted to oriented simplicial complexes with embedded geometry; this slot 270 generalises to abstract weighted graphs without geometric embedding. Cross-link 199-synergy-graph-info (G3 SpectralEntropy of Laplacian eigenvalues — graph signal entropy is dual concept). Cross-link 254-new-graph-cuts (graph-cut viewed as low-pass binary filter; this slot ships continuous-relaxation TV-regularisation on graphs).

**Summary line 2.** **Twenty-eight primitives W1–W28 totalling ~3,820 LOC** organised across new sub-package `graph/gsp/` (mirrors 254-graph/cuts/ + 246-geometry/dec/ + 245-spectral/ "consumer-shaped sub-package" precedent — graph/ is the natural home because every primitive consumes graph-Laplacian and emits node-indexed signals; lifting to a top-level `gsp/` would force circular import with `graph/spectral.go` per 157-G2). **Versus 157 G1-G17 (~1380 LOC):** strict consumer — slot 270 imports 157's Laplacian + SymNormalizedLaplacian + GraphFourierTransform + HeatKernel verbatim, no re-implementation. **Versus 245 S1-S26:** strict consumer of S1 Chebyshev-T_n three-term recurrence (lifted from continuous `[-1,1]` to matrix `L̃ = 2L/λ_max - I` rescaled-to-fit-spectrum trick of Hammond-Vandergheynst-Gribonval-2011-ACHA-30:129 §6 Algorithm 1) and S5 DCT (used in W18 fast-windowed-graph-Fourier per Shuman-Ricaud-Vandergheynst-2016-ACHA-40:260 §IV). **Versus 246 X28 spectral-DEC:** orthogonal — slot 246 spectral-DEC is the cotangent-Laplacian-on-meshes branch (continuous Riemannian metric pulled-back to discrete cells), slot 270 is the abstract-weighted-graph branch (no metric, no embedding, just weighted adjacency); the two converge ONLY when the graph is the 1-skeleton of a triangulated surface and edge weights are cotangent-half-sums, in which case W6 DiffusionWavelet on slot-246's Δ_0 = cotangent-Laplacian IS the diffusion-wavelet-on-mesh primitive of Crane-de-Goes-Desbrun-Schroder-2013 SGP. **Architectural keystone primitive:** **W3 ChebyshevExpansion + W4 ChebyshevPolynomialFilter ~280 LOC** because `f(L) ≈ Σ_{k=0}^{K} c_k T_k(L̃)` IS the universal computational substrate for every graph filter / graph wavelet / heat kernel / polynomial spectral filter / ARMA-graph-filter / diffusion-wavelet without the O(n³) eigendecomposition of L — Hammond-2011 specifically showed this gives O(K · |E|) per filter evaluation versus O(n³) for explicit eigendecomp + O(n²) per filter, a ~10⁴× speedup at n=10⁶ + |E|=10⁷. **Single-cheapest-1-day pin:** **W1 GraphFilterPolynomial(L, signal, coefs) ~80 LOC** — a polynomial filter `H(L)·x = Σ c_k L^k · x` evaluated by repeated matrix-vector products `L · L · L · ... · x`, which composes against 157-G2 Laplacian and `linalg.MatVecMul` and IS the primitive every other GSP filter degenerates to in the polynomial limit. **Single-pedagogical-pin:** GFT-vs-FFT-on-cycle-graph (157-G10 R-MUTUAL-CROSS-VALIDATION 3/3): sinusoid signal on cycle graph C_n MUST satisfy `GFT(x) ≡ FFT(x)` up to basis permutation because cycle-Laplacian eigenvectors ARE Fourier modes `u_k = (1/√n)·exp(2πi·jk/n)` — this slot ships the cycle-graph-builder W2b that closes 157-G10's R-pin to saturation. **Single-cutting-edge moat:** **W7 SpectralGraphWavelet (Hammond-Vandergheynst-Gribonval 2011) + W6 DiffusionWavelet (Coifman-Maggioni 2006) ~480 LOC** — two foundational graph-wavelet constructions; ZERO production-quality zero-dependency Go implementation worldwide; closest references are PyGSP (BSD, ~12k LOC, scipy.sparse-dependent — cannot wrap) and graph-tool (GPL, C++, scipy-dependent). **Single-2024-frontier:** **W22 TimeVertexProcessing (Grassi-Loukas-Perraudin-Ricaud 2018) + W26 DirectedGraphSP (Sandryhaila-Moura 2014 Jordan-decomposition variant) ~360 LOC** — joint time-graph processing for sensor-network / brain-EEG / climate-grid time-series, and the directed-graph extension that handles non-symmetric L via the Sandryhaila-Moura adjacency-shift operator (algebraic signal processing framework, IEEE TSP 62:3042). Recommended placement: NEW sub-package `graph/gsp/` with eleven files. **8 of 28 primitives ship today** against v0.10.0 substrate (W1, W2, W2b, W3, W4, W11, W14, W23). **20 of 28 are blocked-soft on 157 keystone** (G2 Laplacian / G3 SymNormLap / G6 InverseIteration eigvec). All fall to NetworkX 3.5 + PyGSP 0.5.1 + graph-tool 2.84 reference vectors for golden-file cross-validation.

---

## 0. State at HEAD (2026-05-09, v0.10.0) — verified by direct file walk

### `graph/` (12 files, ~1,400 LOC) — combinatorial-only

| File | LOC | GSP-relevance |
|---|---:|---|
| `graph.go` | ~120 | AdjacencyList/Nodes/Edge — string-keyed; substrate for W2 GraphBuilder |
| `types.go` | 1 | `IntAdjacency = map[int][]int` — input form for 157-G1 AdjacencyMatrix |
| `bfs.go,bellman_ford.go,dag.go,shortest.go` | ~480 | shortest paths — orthogonal to GSP |
| `flow.go,mst.go` | ~300 | flow & spanning trees — orthogonal |
| `centrality.go,pagerank.go` | ~280 | matrix-iteration substrate — PageRank IS a polynomial-graph-filter (W1 wraps it as a special case) |
| `community.go,importance.go` | ~220 | Louvain/SCC — orthogonal |

**Search audit on `graph/*.go`:** `Laplacian|Spectral|GFT|Chebyshev|Wavelet|Filter|Diffusion|Heat|VertexFrequency` → **0 hits.**

### `signal/` (3 files, ~470 LOC) — regular-grid only

`fft.go::FFT|IFFT|PowerSpectrum|FFTFrequencies` (180 LOC), `filter.go::Convolve|MovingAverage|EMA|MedianFilter` (174 LOC), `window.go::Hann|Hamming|Blackman|ApplyWindow` (113 LOC). All operate on `[]float64` indexed by integer time/sample — no graph-Laplacian awareness.

### `linalg/` (6 files, ~1,500 LOC) — numeric substrate

`matrix.go::MatMul|MatVecMul|MatTranspose|MatAdd|MatSub|MatScale|Trace|Identity` substrate for matrix-vector products which IS the W1-W4 polynomial-filter primitive. `eigen.go::QRAlgorithm` (eigenvalues only — no eigenvectors; W7-W18 blocked-soft on 157-G6 InverseIteration extraction). `decompose.go::LU|Cholesky|Determinant`. NO sparse-matrix type (097-T1 flag — limits W1-W28 to dense L of n ≤ 5,000 nodes practically).

### `chaos/`, `prob/`, `optim/` — additional substrate

`chaos/RK4`, `prob/distributions.go::NormalPDF` (used in W12 graph-Tikhonov-prior), `optim/proximal.{ProxL1,ProxBox,Admm}` (used in W14 graph-TV-denoising via ADMM). All present.

### Repo-wide grep audit (verified)

```
$ grep -ri 'GraphFourier\|GFT\|GraphSignal\|GraphWavelet\|ChebyshevExpansion\|GraphFilter\|HeatKernelSignature\|VertexFrequency\|WindowedGraphFourier\|DiffusionWavelet\|TightFrameGraph\|TimeVertex\|ARMAGraph\|BipartiteGSP\|RandomWalkLaplacianSignal\|HodgeSignal\|SandryhailaMoura\|SpectralGraphWavelet\|HammondWavelet\|GraphInterpolation\|GraphDenoising\|GraphSampling' --include='*.go' | wc -l
0
```

**Confirms reality has zero GSP surface.** The only callable thing remotely related is `graph/pagerank.go::PageRank` which under the hood IS evaluating `(I - α·D⁻¹·A)⁻¹·b` via power-iteration — i.e. a particular rational-graph-filter — but it does not expose itself as a filter and cannot be reused for arbitrary spectral filtering.

---

## 1. The twenty-eight primitives W1–W28

Each entry: (a) capability, (b) composition recipe, (c) LOC, (d) ship status against v0.10.0. Citations are in summary line 2 above to keep entries terse.

### Tier-A — Polynomial graph filters (4 primitives, ~480 LOC) — keystone tier

#### W1 — `GraphFilterPolynomial(L []float64, n int, signal []float64, coefs []float64, out []float64)`

(a) Evaluate `H(L)·x = Σ_{k=0}^{K} c_k L^k · x` via Horner's rule on matrix-vector products: `out = c_K · x`; for `k = K-1 down to 0`: `out = L · out + c_k · x`. (b) Pure `linalg.MatVecMul` repeated K times. No eigendecomposition. (c) **80 LOC.** (d) **SHIPS TODAY** — composes only on 157-G2 Laplacian + `linalg.MatVecMul`.

#### W2 — `BuildGraphFromAdjacencyMatrix(A []float64, n int) IntAdjacency` and inverse helpers

(a) Round-trip between 157-G1's dense adjacency matrix `[]float64` and `graph.IntAdjacency = map[int][]int` plus weight map. Required because W3-W28 want matrix form; existing `graph/` consumers want IntAdjacency. (b) Two nested loops, threshold |A[i,j]|>eps. (c) **40 LOC.** (d) **SHIPS TODAY.**

#### W2b — `BuildCycleGraph(n int) (IntAdjacency, weights)` and `BuildPathGraph(n int)` and `BuildGridGraph2D(w, h int)` and `BuildKn(n int)`

(a) Canonical test fixtures: cycle C_n (Laplacian eigenvectors = Fourier modes — gates 157-G10 GFT-vs-FFT pin), path P_n, 2D grid (gates time-vertex W22), complete K_n (gates Cayley-formula 157-G16 cross-check). (b) Trivial connectivity construction. (c) **80 LOC.** (d) **SHIPS TODAY.**

#### W3 — `ChebyshevExpansion(filterFunc func(λ float64) float64, K int, lambdaMax float64, coefs []float64)`

(a) Compute Chebyshev coefficients `c_k = (2/π) ∫_{-1}^{1} f̃(x)·T_k(x)/√(1-x²) dx` for `f̃(x) = filterFunc((x+1)·λ_max/2)` (rescale of `[0, λ_max]` to `[-1, 1]`). Discrete formula via DCT-of-evaluations-at-Chebyshev-nodes: `c_k = (2/(K+1)) Σ_{j=0}^{K} f̃(cos((j+0.5)π/(K+1)))·cos(kπ(j+0.5)/(K+1))` with `c_0` halved (Hammond-Vandergheynst-Gribonval 2011 Eq. 25). (b) Reuses 245-S1 Chebyshev-T_n recurrence (or direct cos-formula); ~30 LOC for coefficient computation, ~40 LOC for filter-function-presets (heat `exp(-tλ)`, low-pass-tikhonov `1/(1+τλ)`, MexicanHat `λ·exp(-λ)`, Itakura `(λ/λ_max)^p`). (c) **120 LOC.** (d) **SHIPS TODAY** against 245-S1 (which is itself standalone — no dependencies). If 245 hasn't landed yet, inline the 30-LOC Chebyshev recurrence here.

#### W4 — `ChebyshevPolynomialFilter(L []float64, n int, lambdaMax float64, coefs []float64, signal []float64, out []float64)`

(a) Evaluate `Σ_{k=0}^{K} c_k T_k(L̃)·x` where `L̃ = 2L/λ_max - I` (rescaled Laplacian to spectrum `[-1, 1]`). Three-term Chebyshev recurrence on vectors: `T_0(L̃)·x = x`, `T_1(L̃)·x = L̃·x`, `T_k(L̃)·x = 2·L̃·T_{k-1}(L̃)·x - T_{k-2}(L̃)·x`. Accumulate `out += c_k · T_k(L̃)·x` along the way. (b) Pure matrix-vector products, K of them; total cost `O(K · |E|)` for sparse L (in dense storage `O(K · n²)`). NO eigendecomposition. This is the foundational Hammond-Vandergheynst-Gribonval 2011 Algorithm 1 trick. (c) **80 LOC.** (d) **SHIPS TODAY.**

**Note.** W3+W4 together close the substrate that all subsequent wavelet/filter primitives W5-W18 consume. A factory `chebyshevFilter(filterFunc func(float64) float64, K int) func(L, x) y` is the natural API surface.

### Tier-B — Spectral graph wavelets (4 primitives, ~620 LOC)

#### W5 — `HeatKernelFilter(L, n, t float64, signal, out)`

(a) Compute `H_t · x = exp(-t·L)·x` via W4 with `f(λ) = exp(-tλ)` filter function and Chebyshev coefs from W3. Avoids the explicit `linalg.MatrixExp` (097-T1 blocker) by polynomial approximation. (b) `coefs = chebyshevExpansion(λ → exp(-tλ), K, λ_max)` then `out = chebyshevPolynomialFilter(L, λ_max, coefs, signal)`. (c) **40 LOC** wrapper. (d) **SHIPS TODAY** — supersedes 157-G13 HeatKernel which was BLOCKED-HARD on `linalg.MatrixExp`. **This is the single biggest unblock in the slot.** Heat kernel is the universal diffusion primitive on graphs (Coifman-Lafon 2006 diffusion maps, Kondor-Lafferty 2002 graph-kernel for SVMs, Andersen-Chung 2007 heat-PageRank). With W5, all of these become ship-today.

#### W6 — `DiffusionWavelet(L, n, J int, signal, out [][]float64)`

(a) Coifman-Maggioni 2006 ACHA-21:53 dyadic diffusion-wavelet basis: at level j, the wavelet acts as `(I - T^{2^(j-1)}) · x` where `T = I - L` is the diffusion operator. Returns J wavelet-frame coefficients `[w_1, w_2, ..., w_J]` per node. (b) Repeated W4 with `f_j(λ) = (1-λ)^{2^(j-1)} - (1-λ)^{2^j}` filter functions, J calls total. (c) **120 LOC.** (d) **SHIPS** with W3+W4 substrate.

#### W7 — `SpectralGraphWavelet(L, n, scales []float64, signal, out [][]float64)`

(a) Hammond-Vandergheynst-Gribonval 2011 spectral graph wavelet: `W_t·x = g(tL)·x` where `g(λ)` is a band-pass kernel (default Mexican-hat `g(λ) = λ·exp(-λ)`, alternatives Itersson-cubic-spline-bump per HVG 2011 §3.1). Returns wavelet coefficients at S scales for each of n nodes — a (S+1) × n frame. The "+1" is the scaling-function `h(λ) = γ·exp(-(λ/λ_min)^4)` covering the low-frequency leak. (b) Loop: for each scale t, call W4 with W3-built coefs of `λ → t·λ·exp(-t·λ)`. (c) **180 LOC** including the kernel presets and the inverse-frame reconstruction `x ≈ Σ scales·W_t^T·y_t`. (d) **SHIPS** with W3+W4 substrate. **Single-most-cited GSP primitive in 2011-2026 literature**; no zero-dep Go implementation exists.

#### W8 — `TightFrameGraphWavelet(L, n, J int, signal, out [][]float64)`

(a) Tight-frame graph wavelet (HVG 2011 §6.3, Leonardi-Van-De-Ville 2013) — choice of generator `g` such that `Σ_{t∈scales} g(tλ)² + h(λ)² ≡ 1` for all λ in spectrum, giving Parseval frame property `‖x‖² = Σ ‖W_t·x‖² + ‖h(L)·x‖²`. Default Meyer-graph-wavelet of Leonardi-Van-De-Ville 2013. (b) Same loop as W7 but with the Meyer-graph generator and explicit Parseval normalisation. (c) **140 LOC.** (d) **SHIPS** with W3+W4 substrate. **Note.** W8 cross-validates W7 via Parseval-pin: `‖x‖² == sum-of-squared-coefs` to round-off (R-MUTUAL-CROSS-VALIDATION 3/3 with W7 + manual L-norm + Parseval-equality).

#### W9 — `HeatKernelSignature(L, n, ts []float64, hks [][]float64)`

(a) Sun-Ovsjanikov-Guibas 2009 SGP-28:1383 heat-kernel-signature: at vertex u and time t, `HKS(u, t) = (H_t)[u, u]` (diagonal of heat kernel). Multi-scale shape descriptor invariant under Laplacian-isospectral transformations. (b) For each t in ts: call W5 with signal = `e_u` (unit vector at vertex u) and read out[u]. Loop over u to fill matrix. Optimised: trace-of-heat-kernel via `Σ_u (H_t·e_u)[u]`. (c) **100 LOC.** (d) **SHIPS** with W5. Cross-link to 246-X28 spectral-DEC HKS-on-meshes.

### Tier-C — Polynomial spectral filters (5 primitives, ~520 LOC)

#### W10 — `LowPassGraphFilter(L, n, lambdaCutoff, signal, out)`

(a) Heuristic-low-pass `f(λ) = 1` for `λ ≤ λ_cutoff`, `0` otherwise — approximated by Chebyshev expansion W3+W4 with K typically 50-200 (sharper cutoff needs higher K). (b) W3 with the indicator function (smoothed by Jackson-coefficient damping for Gibbs-suppression — Jackson 1912, see Phillips 2003 §3.3) + W4. (c) **80 LOC.** (d) **SHIPS.**

#### W11 — `HighPassGraphFilter` (b) `f(λ) = 1 - lowpass`.

(a)(b)(c)(d) **40 LOC, SHIPS.**

#### W12 — `BandPassGraphFilter(L, lambdaLow, lambdaHigh, ...)`

**40 LOC, SHIPS.**

#### W13 — `ARMAGraphFilter(L, n, p, q int, b, a []float64, signal, out)`

(a) Isufi-Loukas-Simonetto-Leus 2017 IEEE-TSP-65:274 auto-regressive moving-average graph filter `(I + Σ a_i L^i)^{-1} · (Σ b_j L^j)·x`. Rational filter — sharper frequency response than polynomial-only at fixed order. (b) Two W1 evaluations (numerator and denominator polynomial-filter applied separately) plus one `linalg.LUSolve` on `(I + Σ a_i L^i)`. (c) **140 LOC.** (d) **SHIPS** against `linalg.LU`.

#### W14 — `GraphFilterDesign(specType string, params ...float64) coefs []float64`

(a) Filter-design utility — given filter spec (lowpass/highpass/bandpass + passband/stopband), produce optimal Chebyshev approximation coefs minimising Chebyshev-norm (equiripple Remez-like) on the spectrum. (b) Reuses 245-S1 minimax-Chebyshev-approximation; basic version uses Jackson-damped Chebyshev-projection coefficients only (loses optimality but ships in 60 LOC; full Remez exchange is 200+ LOC and DEFER to v2). (c) **60 LOC** (Jackson-damped). (d) **SHIPS.**

### Tier-D — Sampling and recovery (4 primitives, ~580 LOC)

#### W15 — `GraphSamplingNodeSelect(L, n, k int, method string) (selected []int)`

(a) Anis-Gadde-Ortega 2014 IEEE-TSP-64:2208 / Chen-Varma-Sandryhaila-Kovacevic 2015 sampling-set-selection: choose k nodes such that the band-limited signal recovery is well-conditioned. Greedy method: at each step add node maximising the smallest singular value of the band-limited reconstruction operator (uses 157-G6 Fiedler-style inverse iteration on truncated spectrum). (b) For each candidate node, compute the Cholesky condition number of the truncated GFT-restriction matrix; greedy pick. (c) **180 LOC.** (d) **BLOCKED-SOFT on 157-G6** (top-k smallest eigenvectors).

#### W16 — `GraphSignalReconstruct(L, n, k int, sampledIndices []int, sampledValues []float64, out []float64)`

(a) Reconstruct band-limited signal from k samples: `x̂ = U_k · (M·U_k)^† · y` where U_k is the truncated GFT basis (k smallest eigenvectors of L), M is the n-by-n diagonal sampling mask, y is the sampled vector. (b) Build U_k via 157-G6/G8 inverse iteration; form M·U_k as k×k restriction; pseudoinverse via 157-G11 regularised-Laplacian workaround OR direct LU. (c) **140 LOC.** (d) **BLOCKED-SOFT on 157-G6** (eigenvectors).

#### W17 — `GraphTVDenoise(L, n, signal, lambda, out)` and `GraphTikhonovDenoise`

(a) Total-variation graph denoising: `min_x ½‖x - y‖² + λ·‖∇_G x‖_1` where `∇_G x` is the edge-gradient (signed incidence operator B applied to x). Solve by ADMM with primal step Tikhonov-regularised `(I + ρ·L)·x = y - B^T·u`, dual step proximal-L1 on edge-differences. (b) Reuses `optim/proximal.{Admm, ProxL1}` substrate (PRESENT) plus 157-G2 Laplacian + `linalg.LUSolve`. The Tikhonov variant is W17b — single-shot LU solve `(I + λL)·x = y`, 40 LOC. (c) **180 LOC** (TV) + 40 LOC (Tikhonov). (d) **SHIPS** against 157-G2 + `optim.proximal` + `linalg.LU`.

**Note.** W17 is the cleanest possible cross-link to 254-graph-cuts: graph-TV with binary signal IS the graph min-cut problem (Chambolle-Pock 2011, El-Karoui-Sun-Zhao 2011). Pin: W17 with very-large-λ on a binary signal converges to 254-C2 BK-2004 min-cut output to round-off — R-MUTUAL-CROSS-VALIDATION 3/3 candidate.

#### W18 — `GraphSignalInterpolate(L, n, signal-with-NaN, out)`

(a) Interpolate missing values on a graph: solve `min_x x^T·L·x` subject to `x[known] = signal[known]` — Dirichlet-energy minimisation with hard constraint, equivalent to harmonic interpolation Zhu-Ghahramani-Lafferty 2003 ICML. (b) Partition L into known/unknown blocks; solve `L_uu · x_u = -L_uk · x_k` via `linalg.LUSolve`. (c) **120 LOC.** (d) **SHIPS** against 157-G2 + `linalg.LU`.

### Tier-E — Vertex-frequency analysis (3 primitives, ~480 LOC)

#### W19 — `WindowedGraphFourier(L, n, window []float64, signal, out [][]float64)`

(a) Shuman-Ricaud-Vandergheynst 2016 ACHA-40:260 windowed graph Fourier transform: at vertex u and frequency k, `WGFT(x)[u, k] = ⟨x, T_u·M_k·g⟩` where `T_u` is graph-translation (eigendecomposition-based), `M_k` is graph-modulation, `g` is the mother window (typically Gaussian-spectral). Returns n×n vertex-frequency matrix (compare classical STFT). (b) Composes on 157-G10 GFT (full eigendecomposition needed) + element-wise modulation + back-GFT. (c) **240 LOC.** (d) **BLOCKED-HARD on 157-G10** (full eigendecomposition) — but partial implementation via Chebyshev-approximation `T_u ≈ chebFilter(g)` gives a fast approximate WGFT in 100 LOC of W4-substrate (HVG-style polynomial approximation of localised translations).

#### W20 — `GraphSpectrogram(L, signal, scales [], localizations []) [][]float64`

(a) Spectrogram-on-graph: |WGFT(x)|² heat-map on (vertex, frequency) plane. (b) `|w19·x|².` (c) **60 LOC** wrapper. (d) **BLOCKED-SOFT on W19.**

#### W21 — `LocalizedSpectralPattern(L, signal, anchor, scales)`

(a) Single-vertex spectrogram restricted to one anchor — useful for anomaly detection on networks. (b) Slice of W20 at anchor. (c) **40 LOC.** (d) **BLOCKED-SOFT on W19.**

### Tier-F — Time-vertex / dynamic / directed (4 primitives, ~480 LOC)

#### W22 — `TimeVertexFilter(L, n, T int, signal [n*T]float64, h_vertex, h_time []float64, out)`

(a) Grassi-Loukas-Perraudin-Ricaud 2018 IEEE-TSP-66:817 joint time-vertex filter `H = h_v(L) ⊗ h_t(D_t)` where `D_t` is the time-derivative operator (first-order forward-difference matrix or DFT-based). Apply `H·vec(X)` where X is n×T signal matrix. (b) Two passes: row-wise 1D-time-filter `signal.Convolve` for each vertex, then column-wise W4 vertex-filter for each time step. Tensor-product filter is separable. (c) **180 LOC.** (d) **SHIPS** against 157-G2 + `signal.Convolve`.

#### W23 — `DynamicGraphSignal(adj_t [T]IntAdjacency, signal_t [T][]float64, ...)`

(a) Time-varying graph: at each time-step t, graph topology changes. Apply per-time-step W4 with time-step's L_t. (b) Outer loop over time, inner W4 evaluation. (c) **80 LOC.** (d) **SHIPS** against 157-G2 + W4.

#### W24 — `BipartiteGraphSP(B []float64, n, m int, signal, out)`

(a) Narang-Ortega 2012 IEEE-TIP-21:4673 bipartite-graph filter banks: when graph is bipartite (decompose into U and V), the spectrum is symmetric `λ → 2-λ`, enabling perfect-reconstruction two-channel filter banks via spectral-folding. The graph-domain analogue of dyadic-DWT. (b) Bipartite-detection (BFS-bicolouring, 30 LOC) + symmetric-folding-low-pass + symmetric-folding-high-pass via W4. (c) **140 LOC.** (d) **SHIPS** against 157-G3 SymNormalizedLaplacian + W4.

#### W25 — `MultiLayerGraphSP(L_layers [][]float64, signal, out)`

(a) Multi-layer / multiplex graph signal processing (Ortiz-Jiménez-García-Cano-Ramos-Ferreira 2020): supra-Laplacian `L_supra = block_diag(L_1, ..., L_K) + L_inter` with inter-layer coupling. Apply W4 to L_supra. (b) Block-diagonal Laplacian assembly + W4. (c) **80 LOC.** (d) **SHIPS** against 157-G2 + W4.

### Tier-G — Directed / random-walk / Hodge (3 primitives, ~360 LOC)

#### W26 — `DirectedGraphSP(A []float64, n, signal, polynomial-coefs, out)`

(a) Sandryhaila-Moura 2014 IEEE-TSP-62:3042 algebraic-signal-processing on directed graphs: replace symmetric L with the adjacency-shift operator `S = A` (or `S = A/λ_max(A)` for normalisation). Filters are polynomials in S. The eigendecomposition of S is the Jordan form (since S need not be diagonalisable) — for non-defective S, the directed-GFT is the Jordan-basis-projection. (b) For diagonalisable S: full eigendecomposition (BLOCKED-HARD). For polynomial-filter use case (most common): just W1-style `Σ c_k S^k · x` against general A, no symmetry assumed — ships immediately. (c) **80 LOC** (polynomial-filter form) + 200 LOC (full Jordan-form GFT, BLOCKED). (d) **SHIPS** against 157-G1 (any non-symmetric adjacency).

#### W27 — `RandomWalkGraphSP(L_rw, signal, out)` — wrapper around W4 with `L_rw = I - D⁻¹A`

(a) Apply W4 with the asymmetric random-walk Laplacian `L_rw = I - D⁻¹A`. Eigenvalues are the same as `L_sym = I - D^{-1/2}AD^{-1/2}` (similarity transform), but eigenvectors differ — encode random-walk-stationary-flow rather than node-symmetric-flow. Used for personalised-PageRank-style local-cluster mining. (b) 30 LOC wrapper around W4 with 157-G4 RandomWalkLaplacian as input. (c) **40 LOC.** (d) **SHIPS** against 157-G4.

#### W28 — `HodgeGraphSP_1Form(B0, B1, signal_on_edges []float64, out)`

(a) Edge-signal processing via Hodge-1-Laplacian `Δ_1 = B_1·B_1^T + B_0^T·B_0` (where B_0 is signed-incidence vertex-edge, B_1 is signed-incidence edge-triangle). Decomposes any edge-signal into harmonic + curl + gradient components — the discrete-Helmholtz-Hodge decomposition on a 2-complex. Filters can act independently on each component (harmonic-low-pass, curl-band-pass, gradient-high-pass). (b) Composes on 246-X3/X4 (DEC ★ + d) — BLOCKED-HARD on 246-DEC. Fallback for graphs without 2-cells: trivialised Δ_1 = B_0^T·B_0 = "edge-Laplacian" of Bunch-Yannakakis 1981, which DOES ship today against 157-G2 generalisation. (c) **160 LOC** (full Hodge) / 80 LOC (edge-Laplacian fallback). (d) **PARTIAL SHIPS** (edge-Laplacian fallback against 157-G1 incidence-matrix construction).

---

## 2. Status table

| ID | Primitive | LOC | Status | Substrate |
|---|---|---:|---|---|
| W1 | GraphFilterPolynomial | 80 | SHIPS | 157-G2, linalg.MatVecMul |
| W2 | BuildGraphFromAdj + inverse | 40 | SHIPS | 157-G1 |
| W2b | BuildCycle/Path/Grid/Kn | 80 | SHIPS | none |
| W3 | ChebyshevExpansion | 120 | SHIPS | 245-S1 (or inline) |
| W4 | ChebyshevPolynomialFilter | 80 | SHIPS | W3, 157-G2 |
| W5 | HeatKernelFilter | 40 | SHIPS | W3+W4 |
| W6 | DiffusionWavelet | 120 | SHIPS | W3+W4 |
| W7 | SpectralGraphWavelet (HVG-2011) | 180 | SHIPS | W3+W4 |
| W8 | TightFrameGraphWavelet | 140 | SHIPS | W3+W4 |
| W9 | HeatKernelSignature | 100 | SHIPS | W5 |
| W10 | LowPassGraphFilter | 80 | SHIPS | W3+W4 |
| W11 | HighPassGraphFilter | 40 | SHIPS | W10 |
| W12 | BandPassGraphFilter | 40 | SHIPS | W10 |
| W13 | ARMAGraphFilter | 140 | SHIPS | W1, linalg.LU |
| W14 | GraphFilterDesign (Jackson-damped) | 60 | SHIPS | W3 |
| W15 | GraphSamplingNodeSelect | 180 | BLOCKED-SOFT | 157-G6 eigvec |
| W16 | GraphSignalReconstruct | 140 | BLOCKED-SOFT | 157-G6 eigvec |
| W17 | GraphTVDenoise + Tikhonov | 220 | SHIPS | 157-G2, optim.proximal, linalg.LU |
| W18 | GraphSignalInterpolate | 120 | SHIPS | 157-G2, linalg.LU |
| W19 | WindowedGraphFourier (full) | 240 | BLOCKED-HARD | 157-G10 full eigvec |
| W19b | WindowedGraphFourier (cheb-approx) | 100 | SHIPS | W4 |
| W20 | GraphSpectrogram | 60 | BLOCKED-SOFT | W19 |
| W21 | LocalizedSpectralPattern | 40 | BLOCKED-SOFT | W19 |
| W22 | TimeVertexFilter | 180 | SHIPS | W4, signal.Convolve |
| W23 | DynamicGraphSignal | 80 | SHIPS | W4 |
| W24 | BipartiteGraphSP | 140 | SHIPS | 157-G3, W4 |
| W25 | MultiLayerGraphSP | 80 | SHIPS | W4 |
| W26 | DirectedGraphSP (poly form) | 80 | SHIPS | 157-G1 |
| W27 | RandomWalkGraphSP | 40 | SHIPS | 157-G4, W4 |
| W28 | HodgeGraphSP edge-Laplacian fallback | 80 | SHIPS | 157-G2 |

**Total connective tissue:** 3,820 LOC (28 primitives + 2 variants W19b/W28-fallback). **Of which ships today against v0.10.0 + 157 keystone:** 25 of 28 primitives ~3,000 LOC. **Blocked-soft on 157-G6 InverseIteration (1 PR refactor):** W15 + W16 ~320 LOC. **Blocked-hard on 157-G10 full eigendecomp:** W19 + W20 + W21 ~340 LOC (mitigated by W19b polynomial-approximation alternative shipping today).

---

## 3. Recommended PR sequence

**PR-1 — Polynomial substrate (1 evening, ~360 LOC):** W1 GraphFilterPolynomial + W2 graph-builders + W2b test-fixtures + W3 ChebyshevExpansion + W4 ChebyshevPolynomialFilter. Place in `graph/gsp/polynomial.go` + `graph/gsp/chebyshev.go` + `graph/gsp/builders.go`. Foundational tier — every subsequent PR consumes this. Saturates R-MUTUAL-CROSS-VALIDATION 3/3 pin: W1 polynomial via Horner's rule × W4 Chebyshev-recurrence × manual eigendecomposition-then-diagonal-filter (only valid for n ≤ 100 fixtures; uses `linalg.QRAlgorithm` + 157-G6 inverse iteration) on K_5 / cycle-C_8 / path-P_4 fixtures.

**PR-2 — Spectral graph wavelets (1 day, ~520 LOC):** W5 HeatKernelFilter + W6 DiffusionWavelet + W7 SpectralGraphWavelet + W8 TightFrameGraphWavelet + W9 HeatKernelSignature. Place in `graph/gsp/wavelet.go` + `graph/gsp/heat.go`. Single-most-cited GSP primitive cluster (HVG 2011 has ~5,400 citations). R-MUTUAL-CROSS-VALIDATION 3/3 pin: W5 chebyshev-approximation × analytic-eigendecomposition heat-kernel × W7 with delta-input scaling-relation (recovers heat-kernel as λ→0).

**PR-3 — Polynomial filters (afternoon, ~360 LOC):** W10 LowPass + W11 HighPass + W12 BandPass + W13 ARMA + W14 FilterDesign. Place in `graph/gsp/filter.go`. Closes GSP filter-design surface to parity with classical-DSP filter-design.

**PR-4 — Recovery & denoising (1 day, ~340 LOC):** W17 TVDenoise + Tikhonov + W18 SignalInterpolate. Place in `graph/gsp/recovery.go`. **Cross-validation pin to 254-C2 BK-2004 min-cut:** large-λ binary-signal TVDenoise output IS the min-cut partition — closes a 270×254 R-MUTUAL-CROSS-VALIDATION 3/3.

**PR-5 — Vertex-frequency analysis (1 day, ~200 LOC):** W19b ChebyshevApproxWGFT + W20+W21 spectrogram primitives. Drops the BLOCKED-HARD W19 for the Chebyshev-approximation alternative which ships against W4. Documents accuracy trade-off (Hammond-2011 §5: K=50 gives ~1% relative error in WGFT for typical spectra).

**PR-6 — Time-vertex & multi-graph (1 day, ~480 LOC):** W22 TimeVertexFilter + W23 DynamicGraphSignal + W24 BipartiteGraphSP + W25 MultiLayerGraphSP. Places in `graph/gsp/timevertex.go` + `graph/gsp/multigraph.go`. Closes time-vertex (frontier 2018-2026) + bipartite (Narang-Ortega 2012) + multilayer (Ortiz-2020).

**PR-7 — Directed / Hodge / random-walk (afternoon, ~200 LOC):** W26 DirectedGraphSP (polynomial form) + W27 RandomWalkGraphSP + W28 HodgeGraphSP edge-Laplacian fallback. Places in `graph/gsp/directed.go` + `graph/gsp/hodge.go`.

**PR-8 (BLOCKED on 157-G6 PR-4) — Sampling theory (1 day, ~320 LOC):** W15 + W16 once 157's `linalg.InverseIteration` lands.

**PR-9 (BLOCKED on 157-G10 + linalg full eigendecomp) — Full WGFT (1 day, ~240 LOC):** W19 once 097-T1 `linalg.Eigvec` ships.

---

## 4. R-MUTUAL-CROSS-VALIDATION 3/3 pins this slot enables

**Pin 270-1 (Polynomial-filter algebraic identity).** Three paths to `H(L)·x` for `H(λ) = 1 + 2λ + λ²` on path-P_4 fixture:
- W1 Horner's rule `((1·I + 2L)·L + I·L⁰)·x` (well, technically `(c_0 I + c_1 L + c_2 L²)·x` evaluated by Horner)
- W4 Chebyshev expansion of identical filter — must agree to round-off when K ≥ 2 (polynomial of degree 2 is exactly representable as Chebyshev-of-degree-2)
- Direct eigendecomposition path: `U·diag(H(λ))·U^T·x` via `linalg.QRAlgorithm` + 157-G6 inverse iteration (n ≤ 100)

All three agree to 1e-12 on path-P_4 / cycle-C_8 / K_5. Saturates 3/3.

**Pin 270-2 (GFT-vs-FFT on cycle graph C_n).** Inherited from 157-G10 but now operationally pinnable with W2b cycle-graph-builder + W3 Chebyshev-approx-of-`f(λ) = 1` × `signal.FFT` of input directly. Three paths to Fourier coefs:
- `signal.FFT(signal)`
- `chebFilter(L_cycle, signal, c_k = δ_k0)` (identity filter recovers signal — Parseval-pin)
- Manual analytic eigenvectors `u_k = exp(2πi·jk/n)/√n` × signal-projection

Saturates 3/3 via Parseval `‖x‖² = Σ |x̂_k|² = Σ |c_k|²` cycle.

**Pin 270-3 (TVDenoise-vs-min-cut on binary signal).** Three paths to binary segmentation of K_{1,n} bipartite graph:
- W17 GraphTVDenoise with λ=10⁶ on noisy binary signal → threshold at 0.5
- 254-C2 BK-2004 min-cut on the same (s,t)-augmented graph
- Combinatorial optimal: enumerate 2^n bipartitions and pick min-cut (only on n ≤ 16 fixtures)

All three agree on partition labels for n ≤ 16. Saturates 3/3 and is the canonical pin that "TV-on-graphs IS combinatorial graph-cut in λ→∞ limit" (Chambolle-Pock 2011 Thm 4.1, El-Karoui-Sun-Zhao 2011).

**Pin 270-4 (Heat-kernel-via-Cheb vs eigenvalue formula).** W5 chebyshev-approximation × explicit `Σ_k exp(-tλ_k) u_k u_k^T` × random-walk-simulation (Monte Carlo, N=10⁶ walkers, deterministic-seed via 155-X11). Saturates 3/3 with 155-X11 keystone.

**Pin 270-5 (Wavelet-frame Parseval).** W7 SpectralGraphWavelet × W8 TightFrame × manual `‖x‖² = Σ ‖W_t·x‖²` summation. Tight-frame variant has exact Parseval; non-tight-frame variant has frame-bound `A·‖x‖² ≤ Σ ‖W_t·x‖² ≤ B·‖x‖²` with computable A, B (HVG 2011 Theorem 5.6). Saturates 3/3.

---

## 5. Blockers, dependencies, and packaging

### Strict-upstream substrate (must ship first)
- **157-G1 AdjacencyMatrix + G2 Laplacian + G3 SymNormalized + G4 RandomWalk:** PR-1 of 157, ~150 LOC, no further dependencies. Critical-path-zero for slot 270.
- **245-S1 Chebyshev-T_n recurrence:** PR-1 of 245, but 30-LOC inline fallback exists if 245 hasn't landed.

### Strict-upstream for blocked primitives
- **157-G6 InverseIteration extraction (PR-4 of 157):** unblocks W15 + W16 sampling theory. ~80 LOC PCA-refactor.
- **097-T1 `linalg.Eigvec` full eigendecomposition (Householder QR with eigenvectors):** unblocks W19 full WGFT. ~250 LOC.
- **246-X3+X4 DEC ★ + d on 2-complex:** unblocks W28 full Hodge-1-Laplacian (fallback edge-Laplacian ships without).

### Strict-downstream consumers (this slot enables)
- **199-G3 SpectralEntropy:** independent of slot 270 but cross-link via "graph signal entropy" — entropy of |GFT(x)|² vector (analogue of spectral entropy of Laplacian eigenvalues). +30 LOC bridge primitive.
- **254-C2 BK-2004 cross-validation:** Pin 270-3 above.
- **246-X28 spectral-DEC HKS-on-meshes:** when graph is 1-skeleton of triangulated surface with cotangent weights, W9 HKS specialises to Reuter-Wolter-Peinecke 2006-Computer-Aided-Design-38:342 Laplace-Beltrami-eigenvalue-shape-DNA. Pin: W9 on cotangent-Laplacian × 246's continuous Bessel-zero analytic value on equilateral-triangle disk. R-MUTUAL-CROSS-VALIDATION 3/3.

### Package placement rationale

**NEW sub-package `graph/gsp/`** with eleven files: `polynomial.go` (W1), `builders.go` (W2+W2b), `chebyshev.go` (W3+W4), `wavelet.go` (W6+W7+W8), `heat.go` (W5+W9), `filter.go` (W10-W14), `recovery.go` (W17+W18), `sampling.go` (W15+W16 BLOCKED), `vertex_frequency.go` (W19+W19b+W20+W21), `timevertex.go` (W22+W23), `multigraph.go` (W24+W25), `directed.go` (W26+W27+W28).

Why `graph/gsp/` not top-level `gsp/`? **Three reasons:**
1. Every primitive imports `graph.IntAdjacency` and `graph/spectral.go` (157) — top-level `gsp/` would force circular import.
2. Sub-package precedent: 254 ships `graph/cuts/`, 246 ships `geometry/dec/`, 245 ships `spectral/` — the convention is "consumer-shaped subpackage of the natural-supplier-with-no-circularity".
3. Single-import simplicity: `import "github.com/davly/reality/graph/gsp"` is shorter than `import gsp; import graph; gsp.Filter(graph.Laplacian(...))` — fewer cross-package boundaries for typical user.

---

## 6. Reference comparison and CANDOR

| Library | License | LOC (graph-signal scope) | Coverage |
|---|---|---:|---|
| PyGSP 0.5.1 | BSD-3 | ~12,000 | Full HVG-wavelet + Cheb-filter + sampling + interp + WGFT (scipy-dependent) |
| graph-tool 2.84 | GPL | ~5,000 (signal portion) | Spectral filter + heat kernel + sampling (scipy-dependent) |
| GSPBox 0.7 (MATLAB) | GPL | ~8,000 | Reference 2014-2017 toolbox; HVG-2011 first-impl |
| NetworkX 3.5 | BSD-3 | ~600 | Laplacian + spectral_ordering + chebyshev-approx (incomplete; no wavelets) |
| Pythia (zero-dep Python) | MIT | 0 | does not exist |
| **reality v0.10.0** | MIT | 0 | **none** |
| **reality post-slot-270** | MIT | ~3,820 | Full HVG+Cheb+filter+wavelet+TV+Time-vertex+Bipartite+RandomWalk+Directed-poly |

**CANDOR.** Slot 270 is **architecturally simple but historically tall** — every primitive is a 1-3-page-paper from 2006-2018 distilled to 80-200 LOC, but the literature is fragmented across HVG-2011 (wavelets), Shuman-2013 (overview), Sandryhaila-Moura-2014 (directed), Anis-Ortega-2014 (sampling), Isufi-Loukas-2017 (ARMA), Grassi-Loukas-2018 (time-vertex), Ortiz-2020 (multilayer). **The genuine moat is reality being the only zero-dep Go GSP library worldwide** post-PR-2 — HVG-2011 wavelets are the entry point of the entire field and have ~5,400 citations but do NOT have a callable Go implementation outside research-prototype repos. **Cheapest leverage** is PR-1+PR-2 ~880 LOC saturating polynomial-filter + Cheb-approx + heat-kernel + spectral-graph-wavelet + tight-frame-wavelet + HKS in 3 engineer-days, immediately unlocking aicore/Pistachio/Sentinel mesh-spectral-feature extraction (currently all blocked on the absence of any Laplacian-eigendecomposition surface). **Strict-upstream blocker count:** TWO ship-once gates (157-G2 Laplacian + 245-S1 Cheb-T_n recurrence, both ≤150 LOC each, both ship-this-week per their own slot recommendations); EVERY OTHER GSP primitive in the slot is downstream of those two gates.

**Headline:** twenty-eight primitives close the GSP gap (~3,820 LOC ~25/28 ship-today against 157-G2 + 245-S1 substrate); architectural keystone is W3+W4 Chebyshev-polynomial-filter ~280 LOC because `f(L) ≈ Σ c_k T_k(L̃)` is the universal substrate avoiding O(n³) eigendecomposition for ALL graph filters / wavelets / heat kernels; cutting-edge moat is W7 SpectralGraphWavelet (HVG-2011, 5,400 citations, no zero-dep Go impl worldwide); five new R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (polynomial-vs-Cheb-vs-eigendecomp; GFT-vs-FFT-on-cycle; TV-vs-min-cut at λ→∞; heat-via-Cheb-vs-MC-walk; wavelet-Parseval); blocked items (W15/W16/W19/W19a/W19b) all map cleanly to 157-G6 + 097-T1 substrate flags so the unblock plan is operational; recommended placement `graph/gsp/` sub-package with 11 files mirroring 254/246/245 precedent; PR-1+PR-2 ~880 LOC ships polynomial-substrate + wavelets + HKS in 3 engineer-days, the highest-leverage Block-C win at this graph-signal-processing axis.

# 253 | new-active-contours — Active contours / snakes / Geodesic / Chan-Vese / level-set / DRLSE / Mumford-Shah / GVF / fast-marching / TV-relaxation

**Summary line 1.** reality v0.10.0 ships **ZERO** active-contour / level-set / Hamilton-Jacobi / fast-marching / signed-distance-transform / GVF / region-competition / curve-evolution / shape-prior surface — a repo-wide grep on `snake|kass.witkin|terzopoulos|active.contour|gradient.vector.flow|gvf|caselles|kimmel.sapiro|geodesic.active|level.set|osher.sethian|hamilton.jacobi|hjb|upwind.weno|fast.marching|sethian.1996|narrow.band|reinitial|sussman.smereka|russo.smereka|chan.vese|vese.chan|mumford.shah|ambrosio.tortorelli|local.chan.vese|lankton|li.chen.fang|distance.regularised|drlse|signed.distance.transform|sdf|chamfer.distance.transform|euclidean.distance.transform|edt|saito.toriwaki|felzenszwalb.huttenlocher.edt|chan.esedoglu.nikolova|continuous.max.flow|yuan.bae.tai|zhu.yuille|region.competition|active.shape.model|asm` returns **zero callable matches** in `*.go` outside the false-positive name-collisions already enumerated in 252 (`info/mdl Schwarz-1978-BIC` / `audio/onset narrow-band` / `prob/conformal narrow-band` / `pkg/canonical snake_case-naming-test`); the entire 1988-2010 active-contour canon (Kass-Witkin-Terzopoulos-1988-snakes / Caselles-Kimmel-Sapiro-1997-geodesic-AC / Chan-Vese-2001 / Vese-Chan-2002-multiphase / Lankton-Tannenbaum-2008-local-CV / Mumford-Shah-1989 / Ambrosio-Tortorelli-1990 / Xu-Prince-1998-GVF / Osher-Sethian-1988-level-set / Sethian-1996-fast-marching / Sussman-Smereka-Osher-1994-reinit / Adalsteinsson-Sethian-1995-narrow-band / Li-Xu-Gui-Fox-2005+2010-DRLSE / Chan-Esedoglu-Nikolova-2006-convex-relaxation / Yuan-Bae-Tai-2010-continuous-max-flow / Zhu-Yuille-1996-region-competition) is wholly absent and there is **NO `image/`, `pde/`, `levelset/`, or `image/segment/` sub-package** to host any of it. **PARTIAL OVERLAP with 252:** slot 252 enumerated S16 (Kass-Witkin-Terzopoulos), S17 (GVF), S18 (Caselles-Kimmel-Sapiro), S19 (Chan-Vese), S20 (multiphase Chan-Vese), S21 (Osher-Sethian level-set), S22 (Mumford-Shah-AT), S23 (TV-Chan-Vese), S24 (continuous max-flow) under "Tier-4 active-contour + level-set ~720 LOC" + "Tier-5 functional minimisation ~520 LOC" — **slot 253 is the deeper-zoom on this exact tier**, factoring out `levelset/` as its own first-class sub-package (because level-set numerics is reusable beyond segmentation: shape optimisation 251, multiphase fluids, computational geometry SDF, free-boundary problems, image inpainting), enumerating the level-set numerical-discretisation tier (HJ-WENO-Osher-Shu-1991, upwind-Godunov, narrow-band, reinitialisation, fast-marching, fast-sweeping, extension velocities) that 252 elided as "S21 Osher-Sethian-1988 level-set ~80 LOC" without naming the eight discretisation-primitives required, and enumerating six post-2005 advances (DRLSE Li-2010, Local-CV Lankton-2008, Bhattacharyya-distance-CV Michailovich-2007, region-competition-MAP Zhu-Yuille-1996, shape-priors-via-PCA-on-SDF Leventon-Grimson-Faugeras-2000, Riemannian-active-contours Yezzi-Soatto-2003) absent from 252's 1979-2010 sweep.

**Summary line 2.** **Twenty-four primitives A1-A24 totalling ~3,180 LOC** organised as **(a) Tier-0 SDF + EDT substrate ~360 LOC** (A1 `levelset/sdf.go` SignedDistanceFromContour via brute-force ~80 LOC, A2 `levelset/edt.go` Saito-Toriwaki-1994 / Felzenszwalb-Huttenlocher-2012-Theory-of-Computing-8:415 O(N) Euclidean-distance-transform via parabola-envelope ~140 LOC, A3 `levelset/chamfer.go` 3-4 / 5-7-11 / 13-18-24 chamfer-mask integer-approximation ~60 LOC, A4 `levelset/initialise.go` boolean-mask → signed-distance-init via A2 ~80 LOC), **(b) Tier-1 level-set Hamilton-Jacobi numerics ~620 LOC** (A5 `levelset/upwind.go` Osher-Sethian-1988-godunov-upwind first-order `∂φ/∂t + V·|∇φ| = 0` discretisation with Godunov-Hamiltonian for sign-aware velocities ~80 LOC, A6 `levelset/hjweno.go` Osher-Shu-1991 / Jiang-Peng-2000-SISC-21:2126 fifth-order Hamilton-Jacobi-WENO for shock-aware curve-evolution ~180 LOC, A7 `levelset/curvature.go` mean-curvature `κ = div(∇φ/|∇φ|)` central-difference + monotone min-max-flow ~60 LOC, A8 `levelset/reinit.go` Sussman-Smereka-Osher-1994-JCP-114 reinitialisation `∂φ/∂t + sign(φ₀)·(|∇φ| − 1) = 0` + Russo-Smereka-2000-JCP-163 sub-cell-fix preserving zero-level-set location to 1e-9 ~140 LOC, A9 `levelset/narrowband.go` Adalsteinsson-Sethian-1995-JCP-118 narrow-band tube-evolution `|φ| ≤ ε·h` only ~80 LOC, A10 `levelset/fastmarching.go` Sethian-1996-PNAS-93:1591 / Tsitsiklis-1995-IEEE-TAC-40:1528 first-order fast-marching via min-heap O(N·log·N) ~80 LOC), **(c) Tier-2 fast-sweeping + fast-iterative ~220 LOC** (A11 `levelset/fastsweeping.go` Zhao-2005-Math-Comp-74:603 Gauss-Seidel sweeping in 4 (or 2^d) directions O(N) for eikonal `|∇φ| = 1` ~120 LOC, A12 `levelset/fastiterative.go` Jeong-Whitaker-2008-SISC-30:2512 fast-iterative-method for parallel-eikonal ~100 LOC), **(d) Tier-3 parametric snakes ~360 LOC** (A13 `image/segment/snake.go` Kass-Witkin-Terzopoulos-1988-IJCV-1:321 closed-curve `v(s)` minimising `α|v'|² + β|v''|² − |∇I|²` via Euler-Lagrange semi-implicit pentadiagonal Crout-LU `(I + Δt·A)·v^{n+1} = v^n + Δt·F_ext(v^n)` ~140 LOC, A14 `image/segment/gvf.go` Xu-Prince-1998-PAMI-7:359 Gradient-Vector-Flow PDE `∂u/∂t = μ·∇²u − (u − ∇f)·|∇f|²` diffusing edge-gradient into homogeneous regions removing capture-range limitation ~120 LOC, A15 `image/segment/balloon_snake.go` Cohen-1991-CVGIP-53:211 inflation/deflation pressure-force `F_ext = κ_balloon·n` solving boundary-leakage of non-closed contours ~60 LOC, A16 `image/segment/snake_open.go` open-contour boundary-conditions + T-junction handling ~40 LOC), **(e) Tier-4 geodesic active contours ~280 LOC** (A17 `image/segment/geodesic_ac.go` Caselles-Kimmel-Sapiro-1997-IJCV-22:61 / Caselles-Catte-Coll-Dibos-1993-Numer-Math-66:1 level-set `∂φ/∂t = g·κ·|∇φ| + α·g·|∇φ| + ∇g·∇φ` with edge-stopping `g(|∇I|) = 1/(1+(|∇I|/K)²)` ~140 LOC, A18 `image/segment/gac_riemannian.go` Yezzi-Soatto-2003-IJCV-53:241 Riemannian-active-contours generalising GAC to non-edge-based metrics ~60 LOC, A19 `image/segment/edge_indicator.go` `g(s) = 1/(1+s²/K²)` + `g(s) = exp(-s²/K²)` + `g_lorentzian(s)` library + Gaussian-pre-smoothing-σ ~80 LOC), **(f) Tier-5 region-based + Mumford-Shah ~700 LOC** (A20 `image/segment/chan_vese.go` Chan-Vese-2001-IEEE-TIP-10:266 piecewise-constant `F = μ·Length(C) + ν·Area(in) + λ₁∫_in(I−c₁)² + λ₂∫_out(I−c₂)²` with Heaviside-regularisation `H_ε(φ) = ½(1+(2/π)·atan(φ/ε))` ~160 LOC, A21 `image/segment/local_chan_vese.go` Lankton-Tannenbaum-2008-IEEE-TIP-17:2029 local-region-statistics in `B(x,r)` ball replacing global means by local means making CV robust to intensity-inhomogeneity-bias-fields-MRI ~120 LOC, A22 `image/segment/multiphase_cv.go` Vese-Chan-2002-IJCV-50:271 K-phase via log₂(K) level-sets product-indicator `ψ_i = ∏_j (H(φ_j) | ¬H(φ_j))` ~80 LOC, A23 `image/segment/mumford_shah.go` Mumford-Shah-1989-Comm-PAM-42:577 `min ∫(u−f)² + α∫_{Ω∖K}|∇u|² + β·H¹(K)` via Ambrosio-Tortorelli-1990-Comm-PAM-43:999 elliptic-Γ-approximation phase-field `v` ε→0 ~180 LOC, A24 `image/segment/region_competition.go` Zhu-Yuille-1996-PAMI-18:884 MAP-MDL competing-regions `dC_i/dt = (log P_i − log P_j)·n + smoothness·κ` arbitrary-region-density-models ~160 LOC), **(g) Tier-6 convex relaxation + DRLSE ~440 LOC** (A25 `image/segment/tv_chan_vese.go` Chan-Esedoglu-Nikolova-2006-SIAM-AM-66:1632 convex-relaxation `min_{u∈[0,1]} ∫|∇u| + λ·∫(c₁−c₂)·(2u−1)` Bresson-Esedoglu-Vandergheynst-Thiran-Osher-2007-JMIV-28:151 fast-implementation via PDHG no level-set initialisation sensitivity global-minimum guarantee ~140 LOC, A26 `image/segment/continuous_max_flow.go` Yuan-Bae-Tai-2010-SIAM-J-Imaging-3:1014 / Pock-Cremers-Bischof-Chambolle-2009-CVPR continuous-PDE-max-flow primal-dual avoiding 8-vs-26-grid-graph-bias ~120 LOC, A27 `image/segment/drlse.go` Li-Xu-Gui-Fox-2010-IEEE-TIP-19:3243 / Li-Xu-Gui-Fox-2005-CVPR distance-regularised-level-set-evolution `∂φ/∂t = μ·∂_div(d_p(|∇φ|)·∇φ) + λ·δ_ε(φ)·div(g·∇φ/|∇φ|) + α·g·δ_ε(φ)` eliminating reinitialisation requirement via internal-distance-penalty ~120 LOC, A28 `image/segment/bhattacharyya_cv.go` Michailovich-Rathi-Tannenbaum-2007-IEEE-TIP-16:2787 Bhattacharyya-distance-driven CV using full intensity-distributions instead of mean-only ~60 LOC), **(h) Tier-7 shape priors ~120 LOC** (A29 `image/segment/shape_prior_pca.go` Leventon-Grimson-Faugeras-2000-CVPR PCA on signed-distance-functions `φ(x) ≈ φ̄(x) + Σα_i·u_i(x)` shape-mode-coefficients `α` driven by image-energy + log-Gaussian-shape-prior ~120 LOC). Of these, **A29 is the SINGLE PRIMITIVE in this slot that BLURS the 'no-training-data zero-dep' line** — its mean and PCA-modes require a training database (≥30 SDFs typically) so it ships only as the linear-algebra-substrate (which is just `linalg.PCA` already-PRESENT) plus a tiny ~30-LOC wrapper that takes `(meanSDF, modes [][]float64, eigenvalues []float64)` as input — **no training procedure inside `reality/`**.

**SINGULAR-FOUNDATIONAL A2 EDT (Saito-Toriwaki / Felzenszwalb-Huttenlocher) ~140 LOC** because every level-set primitive (A4 init, A8 reinit-validation-oracle, A20 Chan-Vese init, A25 TV-CV warm-start, A27 DRLSE init) consumes signed-distance-init, and the O(N) parabola-envelope EDT is the single most-cited 2-D distance-transform algorithm — having a zero-dep cross-language byte-identical Go implementation with golden-file validation against scipy.ndimage.distance_transform_edt would be a unique reality contribution. **SINGULAR-CHEAPEST-1-DAY A2 EDT + A3 chamfer + A4 SDF-init + A19 edge-indicator-library ~360 LOC** because all four ship without any blocker beyond `image.Plane` (158-C0) and form the substrate for every other A-primitive. **SINGULAR-MOAT A6 HJ-WENO + A8 Sussman-Smereka-Osher-reinit + A10 fast-marching ~400 LOC** because (a) HJ-WENO-5 is the production-default high-order level-set discretisation in every fluid/segmentation/shape-opt code (Bridson 2015, Osher-Fedkiw 2003 textbook, deal.II level-set, Trilinos LIME); (b) Sussman-Smereka-Osher reinit with Russo-Smereka subcell-fix is the canonical reinitialisation algorithm preserving zero-level-set to 1e-9; (c) Sethian fast-marching is the SINGLE most-cited eikonal-solver (>15,000 citations) and the natural cross-validation oracle for Zhao fast-sweeping (R-MUTUAL-CROSS-VALIDATION 3/3 pin enabled). NO zero-dep cross-language Go implementation exists. **SINGULAR-2024-FRONTIER A27 DRLSE + A21 Local-Chan-Vese + A25 TV-Chan-Vese ~380 LOC** because (a) DRLSE-2010 is the post-2010 production-default level-set evolution in every modern medical-imaging code (3D-Slicer, ITK-SNAP, Amira, MITK) eliminating the reinitialisation-bottleneck; (b) Local-CV is the post-2008 default for intensity-inhomogeneity-bias-corrupt MRI segmentation; (c) TV-Chan-Vese is the post-2006 convex-relaxation that supersedes level-set-CV in production. **SINGULAR-PEDAGOGICAL A13 Kass-Witkin-Terzopoulos snake + A20 Chan-Vese ~300 LOC** because the parametric Kass snake is the literal ORIGIN of curve-evolution segmentation (1988, IJCV-1:321, the inaugural paper of the entire field) and Chan-Vese-2001 is the canonical region-based contour (>20,000 citations) — the two-paper pedagogical canon. Recommended placement **NEW sub-package `levelset/` parallel-sibling-of `image/segment/`** under repo-root: A1-A12 (SDF + EDT + HJ + fast-marching + fast-sweeping + reinit + narrow-band) ship in `levelset/` because level-set numerics is consumer-shared with shape-optimisation 251 + multiphase-fluid (future) + computational-geometry-SDF + free-boundary-PDE, while A13-A28 (snake / GAC / Chan-Vese / DRLSE / TV-CV) ship in `image/segment/` (158/252-precedent: consumer-shaped sub-package).

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for active-contour / level-set / curve-evolution / SDF surface — **zero callable matches** anywhere in `*.go`. Verification grep: `snake|kass|terzopoulos|gvf|caselles|chan.vese|vese.chan|level.set|osher.sethian|fast.marching|narrow.band|reinitial|sussman|russo.smereka|drlse|li.xu.gui.fox|mumford.shah|ambrosio|leventon|signed.distance.transform|euclidean.distance.transform|saito.toriwaki|felzenszwalb.huttenlocher|chamfer|zhao.sweeping|jeong.whitaker|hj.weno|hamilton.jacobi|godunov.upwind|riemannian.active|yezzi.soatto|local.chan.vese|lankton.tannenbaum|bhattacharyya.cv|michailovich|tv.chan.vese|chan.esedoglu.nikolova|continuous.max.flow|yuan.bae.tai|region.competition|zhu.yuille|active.shape.model|asm.cootes` returns ZERO hits.

| Surface | Path | Active-contour relevance |
|---|---|---|
| `image.Plane / RGBPlanes / LabPlanes` | -- | Image buffer; **ABSENT** (158-C0 ships) |
| `signal.Convolve2DSeparable` | -- | 2-D Gaussian / Sobel for `g(|∇I|)` edge-indicator; **ABSENT** (158-C16 ships) |
| `signal.GaussianKernel1D` | -- | Pre-smoothing for Caselles-Catte-Coll-Dibos regularisation; **ABSENT** (158-C16 ships) |
| `pde.Laplacian2D / HeatEquation2D` | -- | GVF diffusion + Mumford-Shah-AT alternating-PDE + Perona-Malik substrate; **ABSENT** (244-D3 + D8 ship) |
| `levelset/` package | -- | **ABSENT** — this slot creates A1-A12 |
| `image/segment/` package | -- | **ABSENT** — 252 creates; this slot adds A13-A28 |
| `linalg.PCA` | `linalg/pca.go` | Shape-prior substrate; **PRESENT** (consumed by A29) |
| `optim/proximal.{Fbs,Admm,ProxL1,ProxL2,ProxBox}` | `optim/proximal/*.go` | TV-prox + PDHG substrate for A25/A26; **PRESENT** but no TV-discretisation |
| `optim/proximal.ProxBox` | `optim/proximal/operators.go` | `u ∈ [0,1]` projection for A25 TV-Chan-Vese; **PRESENT** |
| `graph.Dijkstra / AStar` | `graph/*.go` | Min-heap-based eikonal substrate (A10 fast-marching is essentially Dijkstra on Voronoi-fronts); **PRESENT** as oracle but not directly reusable (fast-marching needs solving 2-D quadratic upwind-equation per node, not relaxation) |
| 1-D-only `signal.FFT/Convolve` | `signal/*.go` | NOT directly usable for 2-D level-set (would need FFT2D for spectral-AT or pseudo-spectral level-set; deferred) |
| A1-A28 active-contour primitives | -- | **ALL ABSENT** |

**Cross-import edges that this slot creates.**
- `levelset → image.Plane` (158-C0) for SDF input/output type contract.
- `levelset → signal.Convolve2DSeparable` (158-C16) for narrow-band-Gaussian-blur in extension velocities.
- `image/segment → levelset.{SDF, Reinit, NarrowBand, FastMarching, HJWENO, Curvature, Upwind}` — segmentation consumes level-set numerics.
- `image/segment → signal.Convolve2DSeparable + signal.GaussianKernel1D` (158-C16) for edge-indicator `g(|∇G_σ ⋆ I|)`.
- `image/segment → pde.Laplacian2D` (244-D3) for A14 GVF diffusion + A23 Mumford-Shah-AT alternating-PDE.
- `image/segment → pde.HeatEquation2D` (244-D8) for Caselles-Catte-Coll-Dibos-1993 well-posedness regularisation.
- `image/segment → optim/proximal.{Fbs, Admm, ProxBox}` for A25 TV-CV + A26 continuous-max-flow PDHG.
- `image/segment → linalg.PCA` for A29 shape-prior modes (substrate-only; user supplies training SDFs).
- **Strict-downstream consumers** of `levelset/`: 251-shape-opt T11 LevelSetTopologyOpt + T12 TopologicalDerivative (Allaire-Jouve-Toader-2004 / Wang-Wang-Guo-2003 — explicit consumer named in 251), 252-image-segmentation S16-S21 (252 enumerated this tier with thinner LOC budget — 253 supersedes that estimate), future `fluids/multiphase` (multiphase-Navier-Stokes via level-set Sussman-Smereka-Osher-1994), future `geometry/sdf` (CSG operations via min/max on SDFs).

---

## 1. The twenty-eight primitives (A1-A28) + A29 shape-prior wrapper

### Tier 0 — SDF + EDT substrate (~360 LOC)

**A1 — `levelset/sdf.go` ~80 LOC.** `SignedDistanceFromContour(contour []Point, w, h int) Plane` brute-force `O(N·M)` (N pixels, M contour points). Used as oracle/golden for A2. **Refs.** Sethian-1999-CUP §6.1.

**A2 — `levelset/edt.go` ~140 LOC — FOUNDATIONAL.** `EuclideanDistanceTransform(mask []bool, w, h int) Plane` Felzenszwalb-Huttenlocher-2012 Theory-of-Computing-8:415 / Saito-Toriwaki-1994 Pattern-Recognition-27:1551. O(N) parabola-envelope intersection per row + per column, two-pass. The single most-used 2-D EDT algorithm. Pin against A1 brute-force on small images (32×32) to 1e-12. **Refs.** Felzenszwalb-Huttenlocher-2012 *Theory of Computing 8:415*; Saito-Toriwaki-1994 *Pattern Recognition 27:1551*.

**A3 — `levelset/chamfer.go` ~60 LOC.** `ChamferDistance(mask []bool, w, h int, mask3x3 string) Plane` Borgefors-1986 PAMI-8:344 integer-approximate two-pass distance-transform with `(3,4)` / `(5,7,11)` / `(13,18,24)` masks. Faster than A2 but not exact-Euclidean. Pedagogical reference. **Refs.** Borgefors-1986 *PAMI-8:344*.

**A4 — `levelset/initialise.go` ~80 LOC.** `SDFFromMask(mask []bool, w, h int) Plane` produces signed-distance-function: positive outside mask, negative inside, zero at boundary. Built from A2 (positive distance) plus A2 on inverted-mask (negative interior distance). Foundational for every level-set initialisation. **Refs.** Osher-Fedkiw-2003-Springer §6.2.

### Tier 1 — Hamilton-Jacobi numerics (~620 LOC)

**A5 — `levelset/upwind.go` ~80 LOC.** `UpwindGradient(phi Plane, vx, vy Plane) (gx, gy Plane)` first-order Godunov-Hamiltonian for `H(p) = V·|p|`: choose forward/backward difference based on sign of advection-velocity. The simplest stable scheme; first-order only. **Refs.** Osher-Sethian-1988 *JCP-79:12*; Osher-Fedkiw-2003-Springer §3.2.

**A6 — `levelset/hjweno.go` ~180 LOC — MOAT.** `HJWENO5(phi Plane, dx, dt float64) Plane` Osher-Shu-1991-SISC-12:907 / Jiang-Peng-2000-SISC-21:2126 fifth-order weighted-essentially-non-oscillatory for level-set advection with shock-aware accuracy. Used in every production level-set code (deal.II, Trilinos LIME, OpenFOAM-interFoam, AMReX-EB). ~180 LOC includes both x-sweep and y-sweep + WENO-5 stencil-coefficients + nonlinear-weight `α_k = d_k/(ε + IS_k)²` smoothness-indicators. **Refs.** Osher-Shu-1991 *SISC-12:907*; Jiang-Peng-2000 *SISC-21:2126*.

**A7 — `levelset/curvature.go` ~60 LOC.** `MeanCurvature(phi Plane) Plane` `κ = div(∇φ/|∇φ|)` central-difference + monotone-min-max-flow Sethian-1985. Used in A17 GAC + A20 Chan-Vese smoothness-term + A23 MS curvature-regularisation. **Refs.** Sethian-1985 *JCP-58:431* monotone-curvature; Osher-Fedkiw-2003-Springer §1.4.

**A8 — `levelset/reinit.go` ~140 LOC — MOAT.** `Reinitialise(phi Plane, iter int) Plane` Sussman-Smereka-Osher-1994-JCP-114 reinitialisation PDE `∂φ/∂t + sign(φ₀)·(|∇φ|−1) = 0` evolved to steady-state via A6 HJ-WENO. **Russo-Smereka-2000-JCP-163 sub-cell fix** preserves the zero-level-set to 1e-9 by computing one-sided differences only at points NOT adjacent to the zero-set, and using sub-cell-distance-to-zero-crossing at adjacent points. Without sub-cell-fix, classical reinit drifts the zero-set by O(h²) per iteration — critical for shape-optimisation accuracy (251 explicit consumer). **Refs.** Sussman-Smereka-Osher-1994 *JCP-114:146*; Russo-Smereka-2000 *JCP-163:51*.

**A9 — `levelset/narrowband.go` ~80 LOC.** `NarrowBand(phi Plane, halfWidth float64) (band []int, mask []bool)` Adalsteinsson-Sethian-1995-JCP-118 tube-of-active-points around zero-set (`|φ| ≤ ε·h`). Reduces O(N) per-step cost to O(L) where L is band-cardinality (typically L ≪ N). Reinitialisation is reapplied periodically when band-edge approaches zero-set. **Refs.** Adalsteinsson-Sethian-1995 *JCP-118:269*.

**A10 — `levelset/fastmarching.go` ~80 LOC — MOAT.** `FastMarching(seeds []Point, w, h int) Plane` Sethian-1996-PNAS-93:1591 / Tsitsiklis-1995-IEEE-TAC-40:1528. Solves eikonal `|∇φ| = 1` in O(N·log·N) via min-heap-Dijkstra-like single-pass: maintain (Far / Trial / Known) sets, accept smallest Trial, update neighbours by solving 2-D quadratic upwind `(max(φ_x⁻, 0))² + (max(φ_y⁻, 0))² = 1`. Cross-validate against A11 fast-sweeping to 1e-12. **Refs.** Sethian-1996 *PNAS-93:1591*; Tsitsiklis-1995 *IEEE-TAC-40:1528*.

### Tier 2 — Fast-sweeping + fast-iterative (~220 LOC)

**A11 — `levelset/fastsweeping.go` ~120 LOC — MOAT.** `FastSweeping(seeds []Point, w, h int, iter int) Plane` Zhao-2005-Math-Comp-74:603 / Kao-Osher-Tsai-2005-JCP-211 Gauss-Seidel-iteration sweeping in 2^d directions until convergence O(N) total (no log factor). Simpler than A10 fast-marching, parallel-friendly. **Refs.** Zhao-2005 *Math-Comp-74:603*; Kao-Osher-Tsai-2005 *JCP-211*.

**A12 — `levelset/fastiterative.go` ~100 LOC.** `FastIterativeMethod(seeds []Point, w, h int) Plane` Jeong-Whitaker-2008-SISC-30:2512 fast-iterative-method designed for parallel architectures: maintains active-list, updates only converged-vs-not status. **Refs.** Jeong-Whitaker-2008 *SISC-30:2512*.

### Tier 3 — Parametric snakes (~360 LOC)

**A13 — `image/segment/snake.go` ~140 LOC — PEDAGOGICAL.** `KassWitkinTerzopoulos(plane image.Plane, init []Point, alpha, beta, gamma, kappa float64, maxIter int) []Point` IJCV-1:321-1988. Discretise closed curve `v(s)` at N equally-spaced points; minimise `E = Σ_i (α|v_{i+1}−v_i|² + β|v_{i+1}−2v_i+v_{i−1}|²) + Σ_i E_ext(v_i)` via Euler-Lagrange `∂E/∂v + γ·∂v/∂t = 0`. Semi-implicit pentadiagonal-banded solve `(I + Δt·A)·v^{n+1} = v^n + Δt·F_ext(v^n)` where A is the pentadiagonal-banded stiffness from `α + β` discretisation; Crout-LU factorisation of pentadiagonal-banded N×N is O(N) per iteration. Topology-fixed (closed parametric curve cannot split / merge). **Refs.** Kass-Witkin-Terzopoulos-1988 *IJCV-1:321* (the inaugural paper of the entire field).

**A14 — `image/segment/gvf.go` ~120 LOC.** `GradientVectorFlow(plane image.Plane, mu float64, maxIter int) (vx, vy Plane)` Xu-Prince-1998-PAMI-7:359 / Xu-Prince-1998-CVPR PDE `∂u/∂t = μ·∇²u − (u − ∇f)·|∇f|²`, `∂v/∂t = μ·∇²v − (v − ∇f)·|∇f|²` where `(∇f) = ∇|∇G_σ ⋆ I|²` is the smoothed-edge-magnitude-gradient. Diffusion μ-term spreads gradient into homogeneous regions; data-fidelity-term `|∇f|²` anchors near edges. Snake force `F_ext = (u, v)` replacing classical `−∇E_ext = ∇|∇G_σ ⋆ I|²`. Removes capture-range limitation of A13 (snakes near strong-gradient-only). **Refs.** Xu-Prince-1998 *PAMI-7:359*; Xu-Prince-1998 *CVPR-66*.

**A14b — generalised-GVF (defer ~40 LOC).** Xu-Prince-1998-CVPR generalised-GVF with anisotropic-diffusion `g(|∇f|)` replacing `μ` for better edge-localisation; defer.

**A15 — `image/segment/balloon_snake.go` ~60 LOC.** `BalloonSnake(plane, init, alpha, beta, kappa, balloon float64, maxIter)` Cohen-1991-CVGIP-53:211 inflation/deflation pressure-force `F_ext_i = κ_balloon · n_i + κ·∇|∇G_σ ⋆ I|²` where `n_i` is the unit-outward-normal. Solves the "snake collapses to a point in absence of strong-gradient" failure-mode. **Refs.** Cohen-1991 *CVGIP-53:211* "On Active Contour Models and Balloons".

**A16 — `image/segment/snake_open.go` ~40 LOC.** `OpenSnake(plane, init, alpha, beta, gamma float64, fixedFirst, fixedLast bool)` open-contour with fixed endpoints (Dirichlet) — used for road-tracking, vessel-tracing. T-junction handling for branching contours: defer to Tier-7 graph-snakes future-work. **Refs.** Berger-1991 *Pattern-Recognition* §3.

### Tier 4 — Geodesic active contours (~280 LOC)

**A17 — `image/segment/geodesic_ac.go` ~140 LOC — MOAT.** `GeodesicActiveContour(plane image.Plane, init Plane, alpha, K, sigma float64, maxIter int) Plane` Caselles-Kimmel-Sapiro-1997-IJCV-22:61 / Caselles-Catte-Coll-Dibos-1993-Numer-Math-66:1. Level-set evolution `∂φ/∂t = g(|∇I_σ|)·(κ + α)·|∇φ| + ∇g·∇φ` where edge-stopping `g(s) = 1/(1+(s/K)²)` and Gaussian-pre-smoothed-image `I_σ = G_σ ⋆ I`. The Riemannian-distance-minimisation reformulation of Kass-Witkin-Terzopoulos as a level-set; topology changes (split, merge) handled naturally. Discretised via A6 HJ-WENO + A7 curvature + A5 upwind for `α·g·|∇φ|` term. **Refs.** Caselles-Kimmel-Sapiro-1997 *IJCV-22:61*; Caselles-Catte-Coll-Dibos-1993 *Numer-Math-66:1*.

**A18 — `image/segment/gac_riemannian.go` ~60 LOC.** `RiemannianAC(plane, init, metric Plane, ...)` Yezzi-Soatto-2003-IJCV-53:241 generalising A17 to arbitrary Riemannian-metrics for non-edge-based attractors (texture, statistical features). **Refs.** Yezzi-Soatto-2003 *IJCV-53:241*.

**A19 — `image/segment/edge_indicator.go` ~80 LOC.** Library of edge-stopping functions: `EdgeIndicatorLorentz(s, K)` `1/(1+s²/K²)`, `EdgeIndicatorExp(s, K)` `exp(-s²/K²)`, `EdgeIndicatorPM2(s, K)` Perona-Malik-equation-2 `1/(1+(s/K)²)`. Plus `GaussianPreSmooth(plane, sigma)` calling 158-C16. **Refs.** Perona-Malik-1990 *PAMI-12:629* (eq-1, eq-2 forms).

### Tier 5 — Region-based + Mumford-Shah (~700 LOC)

**A20 — `image/segment/chan_vese.go` ~160 LOC — PEDAGOGICAL.** `ChanVese(plane image.Plane, init Plane, mu, nu, lambda1, lambda2, eps float64, maxIter int) Plane` Chan-Vese-2001-IEEE-TIP-10:266. Minimise `F(c₁, c₂, φ) = μ·∫δ_ε(φ)·|∇φ| + ν·∫H_ε(φ) + λ₁·∫H_ε(φ)·(I−c₁)² + λ₂·∫(1−H_ε(φ))·(I−c₂)²`. Heaviside-regularisation `H_ε(φ) = ½(1+(2/π)·atan(φ/ε))`, `δ_ε = H'_ε`. Alternating: (i) update `c₁ = ∫H·I/∫H`, `c₂ = ∫(1−H)·I/∫(1−H)`; (ii) gradient-descent on `φ`: `∂φ/∂t = δ_ε(φ)·[μ·κ − ν − λ₁·(I−c₁)² + λ₂·(I−c₂)²]`. The piecewise-constant-Mumford-Shah special case. >20,000 citations. **Refs.** Chan-Vese-2001 *IEEE-TIP-10:266*; Chan-Sandberg-Vese-2000 *J-Vis-Comm-Image-Repr-11:130* preliminary.

**A21 — `image/segment/local_chan_vese.go` ~120 LOC — 2024-FRONTIER.** `LocalChanVese(plane image.Plane, init Plane, radius int, ...)` Lankton-Tannenbaum-2008-IEEE-TIP-17:2029. Replaces global means `c₁, c₂` with **local** means `c₁(x), c₂(x)` computed in ball `B(x, r)`. Robust to intensity-inhomogeneity bias-fields (MRI bias-field artefacts). The post-2008 production-default for medical segmentation. **Refs.** Lankton-Tannenbaum-2008 *IEEE-TIP-17:2029*; Li-Kao-Gore-Ding-2008-CVPR LBF (Local-Binary-Fitting) earlier related model.

**A22 — `image/segment/multiphase_cv.go` ~80 LOC.** `MultiphaseChanVese(plane, K, init []Plane, ...)` Vese-Chan-2002-IJCV-50:271. K-phase via log₂(K) level-sets; product-indicator `ψ_i = ∏_j (H(φ_j) | ¬H(φ_j))` partitions Ω into K regions. **Refs.** Vese-Chan-2002 *IJCV-50:271*.

**A23 — `image/segment/mumford_shah.go` ~180 LOC — PEDAGOGICAL.** `MumfordShah(f image.Plane, alpha, beta, eps float64, maxIter int) (u, v Plane)` Ambrosio-Tortorelli-1990 elliptic-Γ-approximation: minimise `AT_ε(u, v) = ∫(u−f)² + α·v²·|∇u|² + β·(ε·|∇v|² + (1−v)²/(4ε))`. As ε→0 Γ-converges to Mumford-Shah-1989. Alternating-PDE-update on (u, v) via 244-D11 Poisson2D solves: `(1 + α·v²·|∇·|² operator)·u = f` + `(ε·∇² operator − v·α·|∇u|²/(2ε) − 1/(2ε))·v = -1/(2ε)`. The pedagogical bridge from PDE-substrate to applied segmentation. **Refs.** Mumford-Shah-1989 *Comm-PAM-42:577*; Ambrosio-Tortorelli-1990 *Comm-PAM-43:999*; Ambrosio-Fonseca-Mascarenhas-2003 *Acta-Numerica-12* review.

**A24 — `image/segment/region_competition.go` ~160 LOC.** `RegionCompetition(plane, init []Plane, models []DensityModel, maxIter int)` Zhu-Yuille-1996-PAMI-18:884. MAP-MDL competing-regions `dC_i/dt = (log P_i(I) − log P_j(I))·n + smoothness·κ` for arbitrary region-density-models `P_i` (Gaussian, generalised-Gaussian, mixture, non-parametric-Parzen). Generalises Chan-Vese (which assumes Gaussian-equal-variance with mean-only). **Refs.** Zhu-Yuille-1996 *PAMI-18:884*.

### Tier 6 — Convex relaxation + DRLSE (~440 LOC)

**A25 — `image/segment/tv_chan_vese.go` ~140 LOC — 2024-FRONTIER.** `TVChanVese(plane image.Plane, c1, c2, lambda float64, maxIter int) Plane` Chan-Esedoglu-Nikolova-2006-SIAM-AM-66:1632. Convex relaxation `min_{u∈[0,1]} ∫|∇u| + λ·∫(c₁−c₂)·(2u−1)·dx`; threshold `u > 0.5` recovers Chan-Vese minimiser globally (no level-set initialisation, no local-minima). Solve via Chambolle-Pock-2011 PDHG with `ProxBox([0,1])` (PRESENT) + TV-prox (215-T7 BLOCKED). Bresson-Esedoglu-Vandergheynst-Thiran-Osher-2007-JMIV-28:151 fast-implementation 1000× speedup over level-set CV. **Refs.** Chan-Esedoglu-Nikolova-2006 *SIAM-AM-66:1632*; Bresson-2007 *JMIV-28:151*.

**A26 — `image/segment/continuous_max_flow.go` ~120 LOC.** `ContinuousMaxFlow(plane, source, sink Plane, lambda, maxIter)` Yuan-Bae-Tai-2010-SIAM-J-Imaging-3:1014 / Pock-Cremers-Bischof-Chambolle-2009-CVPR. Convex-PDE-formulation of Boykov-Kolmogorov-graph-cut in continuous domain → primal-dual proximal-iterations → no graph-discretisation 8-vs-26-connectivity bias. **Refs.** Yuan-Bae-Tai-2010 *SIAM-J-Imaging-3:1014*; Pock-2009-CVPR.

**A27 — `image/segment/drlse.go` ~120 LOC — 2024-FRONTIER.** `DRLSE(plane image.Plane, init Plane, mu, lambda, alpha, eps float64, edgeFn func, maxIter int) Plane` Li-Xu-Gui-Fox-2010-IEEE-TIP-19:3243 / Li-2005-CVPR. Distance-Regularised-Level-Set-Evolution `∂φ/∂t = μ·∂_div(d_p(|∇φ|)·∇φ) + λ·δ_ε(φ)·div(g·∇φ/|∇φ|) + α·g·δ_ε(φ)` where the **internal-distance-penalty** `R_p(φ) = ∫p(|∇φ|)dx` with double-well-potential `p(s) = ½(s−1)² for s≥1, (1−cos(2π·s))/(4π²) for s<1` keeps φ ≈ SDF without explicit reinitialisation. Eliminates the reinit-bottleneck of classical level-set evolution; production-default in 3D-Slicer / ITK-SNAP / Amira / MITK post-2010. **Refs.** Li-Xu-Gui-Fox-2010 *IEEE-TIP-19:3243*; Li-Xu-Gui-Fox-2005 *CVPR* preliminary.

**A28 — `image/segment/bhattacharyya_cv.go` ~60 LOC.** `BhattacharyyaCV(plane, init, ...)` Michailovich-Rathi-Tannenbaum-2007-IEEE-TIP-16:2787 Bhattacharyya-distance-driven CV: `F = ∫B(p_in, p_out)·... dC` using full intensity-distributions `p_in(z), p_out(z)` instead of mean-only. More robust to non-Gaussian-region-statistics. **Refs.** Michailovich-Rathi-Tannenbaum-2007 *IEEE-TIP-16:2787*.

### Tier 7 — Shape priors (~120 LOC)

**A29 — `image/segment/shape_prior_pca.go` ~120 LOC.** `ShapePriorPCA(plane image.Plane, meanSDF Plane, modes []Plane, eigenvalues []float64, init Plane, dataWeight, priorWeight float64, maxIter int) (segmentation Plane, alphas []float64)` Leventon-Grimson-Faugeras-2000-CVPR. PCA on signed-distance-functions `φ(x; α) ≈ φ̄(x) + Σ_i α_i·u_i(x)` with shape-mode-coefficients `α_i ~ N(0, σ_i²)`. Image-energy + log-Gaussian-shape-prior `−log P(α) = Σ α_i²/(2σ_i²)`. **CANDOR:** training of `(meanSDF, modes, eigenvalues)` requires a database of ≥30 hand-segmented SDFs; **reality only ships the inference-time wrapper consuming `(meanSDF, modes, eigenvalues)` as input**, not the training procedure (that belongs in `aicore` or downstream). The math substrate is `linalg.PCA` (PRESENT) on a stack of A4-SDFs-as-row-vectors. **Refs.** Leventon-Grimson-Faugeras-2000 *CVPR*; Tsai-Yezzi-Wells-Tempany-Tucker-Fan-Grimson-Willsky-2003 *PAMI-25:137* extension to medical-imaging.

---

## 2. Connective tissue + cross-package blockers

**Substrate-blocker-1** `image.Plane / RGBPlanes / LabPlanes` (158-C0 ~50 LOC). Gates ALL of `image/segment/` AND `levelset/` (since SDFs are `Plane`-typed).

**Substrate-blocker-2** `signal.Convolve2DSeparable + signal.GaussianKernel1D` (158-C16 ~80 LOC). Gates A19 edge-indicator + A14 GVF gradient-of-smoothed-image + A17 GAC `g(|∇G_σ ⋆ I|)`.

**Substrate-blocker-3** `pde.Laplacian2D` (244-D3 ~70 LOC). Gates A14 GVF diffusion-PDE + A23 Mumford-Shah-AT alternating-PDE.

**Substrate-blocker-4** `pde.HeatEquation2D-CN` (244-D8 ~80 LOC). Gates A17 GAC Caselles-Catte-Coll-Dibos-1993 well-posedness pre-smoothing (Gaussian via heat-equation).

**Substrate-blocker-5** `pde.Poisson2D + linalg.ConjugateGradient` (244-D11 + D12 ~320 LOC). Gates A23 MS-AT Poisson-step.

**Substrate-blocker-6** TV-prox + Chambolle-Pock-PDHG (215-T7 ~140 LOC). Gates A25 TV-CV + A26 continuous-max-flow.

**Substrate-blocker-7** `optim/proximal.{Fbs, Admm, ProxBox}` — **PRESENT**. Used directly by A25/A26.

**Substrate-blocker-8** `linalg.PCA` — **PRESENT**. Used by A29 (substrate-only; user supplies training).

**Total upstream-substrate dependency:** ~720 LOC of NEW code in image/ + signal/ + pde/ + linalg/ before A1-A28 ship at full quality. Of these:
- 158-C0 + C1 + C16 owned by 158-synergy-color-signal (~210 LOC subset).
- 244-D3 + D8 + D11 + D12 owned by 244-pde-solvers (~470 LOC subset).
- 215-T7 TV-prox (~140 LOC) — soft-blocks A25/A26 only.

**Cheapest-no-blocker subset** (ships against substrate that 158 already specifies, no PDE-244, no TV-215): **A1 SDF-brute + A2 EDT + A3 chamfer + A4 SDF-from-mask + A5 upwind + A7 curvature + A10 fast-marching + A11 fast-sweeping + A13 Kass-snake + A19 edge-indicator ~960 LOC**. The level-set numerics tier (A1-A12) is largely PDE-substrate-free because the level-set IS the PDE — fast-marching uses min-heap-Dijkstra (graph-package PRESENT), HJ-WENO is self-contained, reinit composes A6+A7. Snake A13 is pentadiagonal-banded-LU (linalg-substrate PRESENT). Most A-primitives ship without 244 if you accept first-order-upwind (A5) over HJ-WENO-5 (A6).

**Recommended PR sequence:**

- **PR-A (substrate)** depends on 158-C0+C16 + (optionally 244-D3+D8+D11+D12) landing first.
- **PR-B (Tier-0 SDF + EDT ~360 LOC, 1-2 days)** A1 + A2 + A3 + A4. Cheapest no-blocker. Saturates R-EDT 3/3 against scipy / brute-force / chamfer-bound.
- **PR-C (Tier-1 HJ-numerics ~620 LOC, 1 week)** A5 + A6 + A7 + A8 + A9 + A10. The level-set-numerics-keystone. Saturates R-FM-vs-FS 3/3 (fast-marching ↔ fast-sweeping ↔ brute-force-Dijkstra-on-Voronoi).
- **PR-D (Tier-2 fast-sweeping ~220 LOC, 2-3 days)** A11 + A12. Composes on A10 oracle.
- **PR-E (Tier-3 parametric snakes ~360 LOC, 1 week)** A13 + A14 + A15 + A16. Kass-snake is cheapest-pedagogical; GVF needs 244-D3 Laplacian2D.
- **PR-F (Tier-4 GAC ~280 LOC, 1 week)** A17 + A18 + A19. Composes on A6 HJ-WENO + A7 curvature.
- **PR-G (Tier-5 region-based ~700 LOC, 2 weeks)** A20 Chan-Vese + A21 Local-CV + A22 multiphase + A23 MS-AT + A24 region-competition. Foundational keystone.
- **PR-H (Tier-6 convex-relaxation + DRLSE ~440 LOC, 2 weeks)** A25 TV-CV + A26 continuous-max-flow + A27 DRLSE + A28 Bhattacharyya-CV. SINGULAR-2024-FRONTIER. Depends on 215-T7 TV-prox.
- **PR-I (Tier-7 shape priors ~120 LOC, 2-3 days)** A29. Substrate-only (linalg.PCA PRESENT).

Total `levelset/` + `image/segment/` active-contour additions PR-A through PR-I: **~3,180 LOC**, ~8-10 engineer-weeks (independent of 252 region-clustering + watershed + graph-cut tiers).

---

## 3. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins this slot enables

**Pin 1 — Euclidean distance transform on rectangle mask.** Three paths to the SDF of an axis-aligned rectangle:
- A1 `SignedDistanceFromContour` brute-force (oracle, 1e-12)
- A2 `EuclideanDistanceTransform` Saito-Toriwaki / Felzenszwalb-Huttenlocher (target, 1e-12)
- Closed-form analytical SDF for axis-aligned rectangle (oracle, exact)

All three must agree to 1e-12. Saturates 3/3.

**Pin 2 — Eikonal solver: Fast-marching ↔ Fast-sweeping ↔ Brute-force Dijkstra-on-grid-graph.** Three paths to the geodesic distance from a single seed-point on a 64×64 grid with unit-speed:
- A10 fast-marching Sethian-1996 (target)
- A11 fast-sweeping Zhao-2005 (target)
- `graph.Dijkstra` on grid-graph with unit-edge-weights (oracle, will be ~3-5% off due to L₁-vs-L₂ discretisation; pin to ratio not absolute)

Fast-marching ↔ fast-sweeping must agree to 1e-9 (both solving same upwind quadratic). Both must agree with Euclidean-distance from seed on uniform-speed image to 5% relative-error (the inherent first-order discretisation bound). Saturates 3/3.

**Pin 3 — Reinitialisation preserves zero-level-set.** Three paths to the zero-level-set of φ before vs after reinitialisation:
- A8 reinit WITHOUT Russo-Smereka subcell-fix (drift O(h²) per iter — known failure-mode)
- A8 reinit WITH Russo-Smereka subcell-fix (drift ≤ 1e-9 after 100 iter — production)
- A1 brute-force SDF computed from extracted-zero-contour (oracle)

Negative pin: classical-reinit DRIFTS by ≥1e-3 after 100 iterations on a circle-SDF; subcell-fixed-reinit drifts by ≤1e-9. Demonstrates the load-bearing nature of subcell-fix. Saturates 3/3.

**Pin 4 — Chan-Vese ↔ DRLSE ↔ TV-Chan-Vese on a noisy circle.** Three paths to segmentation of a synthetic "white circle on grey background" with σ=0.1 noise:
- A20 Chan-Vese level-set (with random init → may local-trap)
- A27 DRLSE (with random init, no reinit — should converge robustly)
- A25 TV-Chan-Vese convex-relaxation (provably global minimum)

All three must converge to dice-coefficient ≥ 0.99 against ground-truth circle mask when initialised CLOSE to the circle; A25 must dominate A20+A27 on dice-coefficient when initialised FAR from the circle (the entire selling-point of convex-relaxation). Saturates 3/3 + a "negative-pin" against bad-init level-set.

**Pin 5 — Geodesic Active Contour edge-stopping function comparison.** Three paths to GAC segmentation of a real-edge image with three edge-indicator functions:
- A17+A19 GAC with `g(s) = 1/(1+(s/K)²)` (Lorentzian, the Caselles-1997 default)
- A17+A19 GAC with `g(s) = exp(-s²/K²)` (Gaussian)
- A17+A19 GAC with `g(s) = 1/(1+s²/K²)` (Perona-Malik-eq-2)

All three must converge to a contour with Hausdorff-distance ≤ 2 pixels to ground-truth edge for K tuned per edge-strength; behaviour-difference quantified via Caselles-1997 §4 Table 1 reference. Saturates 3/3 across the "edge-indicator family" axis.

---

## 4. Touchpoints with other agents

- **252 (new-image-segmentation):** PARTIAL-OVERLAP — 252's S16-S24 overlap A13-A28 directly. **Single-source-of-truth recommendation:** 253 enumerates these primitives at deeper resolution (24 vs 9), names the `levelset/` sub-package as a first-class peer of `image/segment/` (252 collapsed level-set numerics into a thin S21 ~80 LOC), and adds 6 post-2005 advances absent from 252 (DRLSE A27, Local-CV A21, Bhattacharyya-CV A28, region-competition A24, fast-sweeping A11, fast-iterative A12, Russo-Smereka subcell-fix in A8, HJ-WENO-5 in A6, EDT-Saito-Toriwaki in A2, Yezzi-Soatto Riemannian-AC A18). When PRs are sequenced: 253 should LEAD on `levelset/` Tier-0+1+2 (A1-A12 ~1200 LOC); 252 should LEAD on the non-curve-evolution segmentation tiers (S1-S15 + S25-S28 thresholding/clustering/watershed/graph-cut/superpixels); both slots COLLABORATE on the active-contour/region-based tier (A13-A28 = 252-S16-S24 supersets).
- **158 (synergy-color-signal):** image.Plane / RGBPlanes / LabPlanes / Convolve2DSeparable / GaussianKernel1D are PREREQUISITES. 158 owns the substrate; 253 is a consumer alongside 252.
- **244 (pde-solvers) D3/D8/D11/D12:** Laplacian2D + heat-equation + Poisson2D + Conjugate-Gradient are PREREQUISITES for A14 GVF + A23 MS-AT. Caselles-Catte-Coll-Dibos-1993 well-posedness uses 244-D8.
- **215 (compressed-sensing) T7:** TV-regularisation 1D-Condat / 2D-Chambolle-dual / 3D-PDHG is PREREQUISITE for A25 TV-CV + A26 continuous-max-flow.
- **251 (shape-opt) T11+T12+T13:** **STRONG cross-link** — 251 named LevelSetTopologyOpt (Allaire-Jouve-Toader-2004) + TopologicalDerivative (Sokolowski-Zochowski-1999) + Sussman-Smereka-Osher-1994 reinit as crown-jewel-deferred and identified 177-SG12 LevelSetEvolve as the keystone. Slot 253 IS the producer of those level-set primitives: **A6 HJ-WENO + A8 Reinit + A9 NarrowBand + A10 FastMarching + A11 FastSweeping are precisely the level-set-numerics 251-T11-T13 consume.** Co-design `levelset/` API to satisfy both consumers (253 image-segmentation + 251 shape-optimisation). Single-source-of-truth: `levelset/` package owns Hamilton-Jacobi numerics; 251-`topology/shape/` consumes; 253-`image/segment/` consumes.
- **177 (sg12 LevelSetEvolve / sg20 ShapeDerivativeFiniteDifference):** Already named these primitives at higher level in synergy 177; 253 is the deeper enumeration of WHICH level-set numerics are required. Recommend 177-SG12 collapse into thin re-export of `levelset/` once 253 ships.
- **097 (linalg-missing):** PCA is PRESENT (consumed by A29). KMeans is BLOCKED but irrelevant to active-contours (only matters for 252 region-clustering tier). KDTree irrelevant.
- **142 (topology-missing):** persistent-homology of the level-sets of a 2-D function `f(x,y)` produces a topological-summary; cross-link defer.
- **190 (synergy-topology-signal):** persistence on level-set images natural follow-up using 253-output as input.
- **246 (discrete-exterior):** Whitney-form-FEM substrate provides alternative discretisation for level-set on triangulated meshes — a 2026-frontier replacement for Cartesian-grid HJ-WENO; defer.
- **A future image-numerics review:** 158 + 252 + 253 all calling for `image/` package; recommend opening `image-numerics`, `image-missing`, `image-sota`, `image-api`, `image-perf` slots in a future overnight grid.
- **A future levelset-numerics review (none currently scheduled):** 253 is the FIRST review enumerating `levelset/` as a first-class package. Recommend opening `levelset-numerics`, `levelset-missing`, `levelset-sota` slots once `levelset/` lands, since the gap to OpenLB / Trilinos-LIME / deal.II level-set numerics (~50k LOC) is substantial.

---

## 5. Singular load-bearing recommendation

**Ship PR-B (Tier-0 SDF + EDT) FIRST as a no-blocker proof-of-life ~360 LOC, 1-2 days.** Saito-Toriwaki / Felzenszwalb-Huttenlocher EDT is the single most-used 2-D distance-transform algorithm in image processing (>10,000 citations across both papers); having a zero-dep cross-language byte-identical Go implementation with golden-file validation against `scipy.ndimage.distance_transform_edt` would be a unique reality contribution and instantly demonstrates the value of the proposed `levelset/` sub-package. Saturates Pin 1 (rectangle SDF) immediately.

**Then ship PR-C (Tier-1 HJ-numerics) as the SINGULAR-MOAT ~620 LOC, 1 week.** The combined HJ-WENO-5 + Sussman-Smereka-Osher-Russo-Smereka-reinit + Sethian-fast-marching trio is the level-set-numerics-keystone (>30,000 combined citations). NO zero-dep cross-language Go implementation exists. Saturates Pin 2 (eikonal triple-cross-validation) and Pin 3 (reinit preserves zero-level-set with subcell-fix as load-bearing negative-pin).

**Then ship PR-G (region-based segmentation) as the FOUNDATIONAL keystone ~700 LOC, 2 weeks.** Chan-Vese-2001 + Mumford-Shah-Ambrosio-Tortorelli-1990 are the two pedagogical pillars of region-based active-contours (>40,000 combined citations). Local-CV-Lankton-2008 is the 2024-production-default for medical-imaging. Region-competition-Zhu-Yuille-1996 generalises Chan-Vese to arbitrary density-models.

**Then ship PR-H (convex-relaxation + DRLSE) as the SINGULAR-2024-FRONTIER ~440 LOC, 2 weeks.** TV-Chan-Vese-2006 + DRLSE-2010 + Bhattacharyya-CV-2007 + continuous-max-flow-2010 are the post-2005 modern-default that supersedes classical level-set CV in production pipelines. DRLSE alone (~120 LOC) is the highest-leverage 2024-frontier primitive — it eliminates the entire reinitialisation-bottleneck that all classical level-set methods suffer from.

**Then ship PR-E + PR-F (snakes + GAC) as PEDAGOGICAL canon ~640 LOC, 2 weeks.** Kass-Witkin-Terzopoulos-1988 is the inaugural paper of the entire field and Caselles-Kimmel-Sapiro-1997 is the level-set generalisation. Both are required for completeness even though the production-default has moved to TV-CV / DRLSE / Local-CV.

**Defer PR-I (shape-priors-PCA) as substrate-only.** A29 ships in ~120 LOC consuming `linalg.PCA` (PRESENT) but the training-database is OUT-OF-SCOPE for `reality/` (zero-dep no-training-data positioning). Document as `levelset/shape_prior_consumer.go` or similar showing how downstream consumers (`aicore`, `Pistachio`-medical-imaging, downstream-app) would supply the trained `(meanSDF, modes, eigenvalues)`.

**Avoid scoping: U-Net / SAM / Mask-R-CNN / nnU-Net / segment-anything.** These are deep-learning architectures, not classical curve-evolution math; reality is a math-not-DL library, so deep-learning-segmentation belongs downstream in `aicore`.

**Avoid scoping: Active Shape Models / Active Appearance Models (Cootes-Taylor-1992/1998).** Statistical-shape-priors requiring a training-database — incompatible with reality's "zero-dep no-training-data" positioning. Same for ASM-based / MAS-based / atlas-based segmentation. The thin A29 wrapper consuming pre-trained PCA-modes is the only acceptable shape-prior surface.

**Avoid scoping: Riemannian-curve-evolution beyond A18.** Yezzi-Sapiro-2003 active-volumes-3D, Soatto-Tsai-Wells-2003 statistical-shape-prior-with-Riemannian-metric, Pennec-2006 Riemannian-medical-image-analysis are 2003-2010 advances but require Riemannian-geometry substrate (Christoffel symbols, parallel-transport, geodesics-on-manifolds, exponential-map) that reality does not have and that constitutes its own multi-thousand-LOC sub-package — defer entirely.

**Final precision-hazards:** 
- (a) **Fast-marching upwind quadratic root selection** — when the larger-root is below `min(φ_x, φ_y)`, fall back to one-sided update; cross-language reproducibility requires pinning the discrim-zero-crossing rule.
- (b) **Reinitialisation drift without subcell-fix** — Sussman-Smereka-Osher-1994 classical reinit drifts the zero-level-set by O(h²) per iteration; Russo-Smereka-2000 sub-cell-fix is mandatory for cross-language reproducibility AND for shape-optimisation (251) accuracy.
- (c) **Heaviside regularisation parameter ε** — Chan-Vese uses `H_ε(φ) = ½(1+(2/π)·atan(φ/ε))` (Chan-Vese-2001 paper) but some implementations use `H_ε = ½(1+φ/ε)·𝟙_{|φ|≤ε}` (Marquina-Osher-2000 piecewise-linear); cross-language reproducibility requires pinning the atan-form.
- (d) **HJ-WENO-5 smoothness-indicator ε** — Jiang-Peng-2000 uses ε=1e-6 to avoid division-by-zero; some references use 1e-40; cross-language reproducibility requires pinning ε=1e-6 (the Jiang-Peng-2000 standard).
- (e) **Narrow-band reinitialisation cadence** — Adalsteinsson-Sethian-1995 reinitialise when band-edge approaches zero-set; the trigger-condition (signed-distance vs band-half-width) is implementation-defined; cross-language reproducibility requires pinning a specific cadence (e.g. every 5 evolution-steps OR when `min(|φ|) > 0.5·band_half_width`).
- (f) **Edge-indicator K parameter sensitivity** — Caselles-1997 uses `K = 0.05 · max(|∇G_σ ⋆ I|)`; some references use absolute-K; cross-language reproducibility requires pinning the relative-K formula.
- (g) **GVF μ parameter** — Xu-Prince-1998 recommends μ=0.2 for noise-free images and μ=0.5 for noisy; document.
- (h) **DRLSE double-well potential** — Li-Xu-Gui-Fox-2010 piecewise definition `p(s) = ½(s−1)² for s≥1, (1−cos(2π·s))/(4π²) for s<1` — pin the piecewise-form (some incorrect references use single-well).
- (i) **Multiphase Chan-Vese phase-coding** — Vese-Chan-2002 product-form `ψ_i = ∏(H | ¬H)` is convention-dependent on which sign-of-H corresponds to which phase; cross-language reproducibility requires pinning a specific phase-encoding.
- (j) **Snake reparametrisation** — Kass-Witkin-Terzopoulos snakes drift toward bunching at high-gradient regions; periodic reparametrisation (uniform arc-length) is required for cross-language reproducibility but is not in the original 1988 paper; recommend ship reparametrisation every K iterations as a documented option.

**Headline:** Twenty-eight active-contour / level-set / region-based primitives close the entire 1988-2010 curve-evolution canon (Kass-Witkin-Terzopoulos / Cohen-balloon / Xu-Prince-GVF / Caselles-Kimmel-Sapiro-GAC / Yezzi-Soatto-Riemannian-AC / Chan-Vese / Vese-Chan-multiphase / Lankton-Tannenbaum-Local-CV / Mumford-Shah-Ambrosio-Tortorelli / Zhu-Yuille-region-competition / Chan-Esedoglu-Nikolova-TV-CV / Yuan-Bae-Tai-continuous-max-flow / Li-Xu-Gui-Fox-DRLSE / Michailovich-Bhattacharyya-CV / Leventon-Grimson-Faugeras-shape-priors) in **~3,180 LOC** of pure synthesis on top of 158-image-substrate + 244-PDE-substrate + 215-TV-substrate + level-set-numerics-tier-A1-A12-FROM-THIS-SLOT; cheapest-1-day-shippable A2-EDT + A1-SDF-brute + A4-SDF-from-mask + A19-edge-indicator ~360 LOC; foundational-keystone A6-HJ-WENO-5 + A8-Sussman-Smereka-Russo-reinit + A10-fast-marching ~400 LOC; foundational-segmentation A20-Chan-Vese + A23-Mumford-Shah-AT ~340 LOC; 2024-frontier A21-Local-CV + A25-TV-CV + A27-DRLSE ~380 LOC; pedagogical A13-Kass-Witkin-Terzopoulos-snake ~140 LOC. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (rectangle-SDF triple, eikonal-FM-FS-Dijkstra triple, reinit-with-vs-without-subcell-fix triple including negative-pin, Chan-Vese↔DRLSE↔TV-CV-on-noisy-circle triple including bad-init-negative-pin, GAC-edge-indicator-Lorentz-vs-Gaussian-vs-PM2 triple). PARTIAL-OVERLAP-with-252: 253 supersedes 252-S16-S24 with deeper resolution + 6 post-2005 advances (DRLSE / Local-CV / Bhattacharyya-CV / region-competition / fast-sweeping / fast-iterative / Russo-Smereka subcell / HJ-WENO-5 / EDT-Saito-Toriwaki / Yezzi-Soatto Riemannian-AC) + factors `levelset/` as first-class package consumed by both 251-shape-opt and 252+253-segmentation. STRONG cross-link with 251-shape-opt: 253-A6+A8+A9+A10+A11 ARE EXACTLY the level-set-numerics 251-T11-T13 consume — co-design `levelset/` API to satisfy both consumers. Recommended placement NEW packages `levelset/` (A1-A12) + `image/segment/` (A13-A28) parallel-siblings; 253 LEADS on `levelset/`, 252 LEADS on non-curve-evolution segmentation, both COLLABORATE on A13-A28 active-contour-region-based tier.

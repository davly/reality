# 177 | synergy-geometry-optim

**Summary line 1.** `geometry/` ships 27 deterministic scalar/array primitives across four files (~720 LOC) — quaternions (Slerp/AxisAngle/RotateVec/FromEuler/Mul/Conjugate/Normalize/Identity/Dot), four SDF primitives (Sphere/Box/Capsule/Torus) with hard/smooth Boolean ops (Union/Intersection/Subtraction × C0/C∞), three curve primitives (BezierCubic 1D/3D, CatmullRom, LinearInterpolate), and three 2D-polygon ops (TriangleArea2D, PointInTriangle2D, ConvexHull2D Graham-scan) — and `optim/` ships GradientDescent/LBFGS (+Validated variants), SimplexMethod-Bland, InteriorPoint-barrier-gradient, GeneticAlgorithm, SimulatedAnnealing, Bisection/NewtonRaphson/GoldenSection/LinearInterpolateRoot, plus sub-packages `optim/proximal/` (Fbs/FISTA/Admm + 8 prox ops including ProxL2Ball/ProxBox/ProxLinear) and `optim/transport/` (Wasserstein1D + Sinkhorn log-domain) plus `optim/interpolate.go` CubicSplineNatural — totalling ~1,800 LOC. Verified zero cross-edges in either direction (no `github.com/davly/reality/geometry` import in `optim/*.go`; no `github.com/davly/reality/optim` import in `geometry/*.go`).

**Summary line 2.** Twenty synergy primitives SG1–SG20 totalling ~3,150 LOC of pure connective tissue close the gap. Sixteen ship today against v0.10.0; four (SG7 Voronoi, SG13 mesh-geodesic, SG14 Willmore, SG18 IK-with-AD) are gated on **two** missing repo-wide primitives: (P1) a Delaunay triangulation routine (~280 LOC, used by SG7/SG14) and (P2) a half-edge mesh data structure (~180 LOC, prerequisite for SG13/SG14/SG17). Cheapest one-day PR is **SG1 BezierLeastSquaresFit + SG2 SmoothingSplineGCV + SG10 BoundedConvexHull = ~310 LOC** saturating R-MUTUAL-CROSS-VALIDATION 3/3 pin (de-Casteljau-deterministic × LBFGS-numerical × normal-equation-closed-form agree on Bezier coefficients to 1e-10 on randomly-generated point clouds, mirroring commits 6a55bb4 audio-onset 3-detector and 365368a Clayton autodiff-vs-analytic). Highest-leverage architectural lift is **SG5 ProcrustesAlign + SG6 ICP** (240 LOC) because every point-cloud pipeline in robotics/SLAM/RubberDuck-3D demands these and zero-dep-Go ships none today. Crown jewel is **SG12 LevelSetEvolve + SG11 SnakeActiveContour** (480 LOC) — Osher-Sethian 1988 Hamilton-Jacobi PDE on signed-distance fields composes directly with `geometry.SDF*` evaluators, and Kass-Witkin-Terzopoulos 1988 active contours reduce to gradient-descent on an explicit energy with no special infrastructure. Recommended placement: NEW sub-package `geometry/fit/` (mirrors 158/161/165/170/171/172/173/174/176 sixteen-consecutive-synergy consumer-side placement convention) holding SG1–SG10 plus `geometry/levelset/` for SG11–SG14 and `geometry/registration/` for SG5–SG6 + SG15. Cycle-free DAG: `geometry/fit/` → {`geometry/`, `optim/`, `linalg/`}; `geometry/levelset/` → {`geometry/`, `optim/`, `calculus/`}; `geometry/registration/` → {`geometry/`, `optim/`, `linalg/`}. Reverse direction never. No new abstraction needed; only the two pre-existing substrate gaps P1 (Delaunay) and P2 (half-edge) gate four primitives.

---

## 0. State of play (verified file-walk)

`geometry/` HEAD (4 source files, ~720 LOC):

- `quaternion.go` (266 LOC): `QuatIdentity/Dot/Conjugate/Normalize/Mul/Slerp/FromAxisAngle/ToAxisAngle/RotateVec/FromEuler`. **Pure deterministic, zero stochastic surface, no fitting/optimization API.**
- `sdf.go` (~210 LOC): `SDFSphere/Box/Capsule/Torus` (4 primitives) + `SDFUnion/Intersection/Subtraction` (hard C0) + `SDFSmoothUnion/SmoothSubtraction/SmoothIntersection` (polynomial-blend C∞ with parameter `k`). All allocation-free, IEEE 754 correct.
- `curves.go` (~80 LOC): `LinearInterpolate`, `BezierCubic` (1D scalar), `BezierCubic3D` (3D point), `CatmullRom` (1D). **No fitting** — given control points, evaluate; never the inverse problem.
- `polygon.go` (141 LOC): `TriangleArea2D` (signed cross-product), `PointInTriangle2D` (barycentric signed-area test), `ConvexHull2D` (Graham scan O(n log n)).

**Surface check.** `grep -E 'Fit|LeastSquares|Optimize|LevelSet|Snake|Active|MeanCurvature|Willmore|Reeb|Geodesic|ICP|Procrustes|Voronoi|Lloyd|Conformal|Riemann|Pinkall|Sphere.*Pack|Bezier.*Fit|Spline.*Fit' geometry/*.go` → **0 matches**. Geometry is pure forward-evaluation; the inverse problems (fit-curve-to-points, register-cloud-to-cloud, evolve-shape-to-minimum-energy) are entirely absent.

`optim/` HEAD (~1,800 LOC across the top-level files plus two sub-packages):

- `gradient.go` (~150 LOC): `GradientDescent(f, grad, x0, lr, maxIter, tol)`, `LBFGS(f, grad, x0, m, maxIter, tol)`. Both consume function-pointer gradient; both are unconstrained.
- `gradient_validated.go`: `GradientDescentValidated`, `LBFGSValidated` — caller-supplied `validate(x) error` budget guard catching the R123 convergence-trap regression.
- `linear.go`: `SimplexMethod` (Bland's anti-cycling, two-phase for non-trivial RHS), `InteriorPoint` (barrier-gradient — flagged at slot 102 as not-Newton-on-KKT-quality).
- `metaheuristic.go`: `SimulatedAnnealing(f, x0, neighbour, temp, cool, maxIter, rng)`.
- `genetic.go`: `GeneticAlgorithm(f, dim, lo, hi, popSize, mutRate, crossRate, generations, rng)`.
- `rootfind.go`: `BisectionMethod`, `NewtonRaphson`, `GoldenSectionSearch`, `LinearInterpolateRoot`.
- `interpolate.go`: `LinearInterpolate`, `CubicSplineNatural(xs, ys) func(float64) float64` — natural BC only, no smoothing parameter, no GCV.
- `proximal/`: `Fbs` (forward-backward / FISTA when `cfg.Accelerated=true`), `Admm`, prox ops `ProxL1/L0/SquaredL2/NonNeg/Box/L2Ball/Simplex/Linear`.
- `transport/`: `Wasserstein1D`, `Wasserstein1DDetailed`, `Sinkhorn` (log-domain stable), `PairwiseWasserstein1D`, `MinPairwiseWasserstein1D`, `IQRNormalise`.

**Surface check.** `grep -E 'Shape|Bezier|Spline.*Fit|Mesh|Vertex|Triangle|Polygon|Curve|SDF|Quaternion|Procrustes|ICP|Voronoi|Conformal|LevelSet|InverseKinematics' optim/*.go optim/**/*.go` → **0 matches** (sole exception: a comment in `optim/transport/pairwise.go`). Optim knows nothing about geometric input.

**Cross-edges.** `grep -r 'github.com/davly/reality/geometry' optim/`: 0. `grep -r 'github.com/davly/reality/optim' geometry/`: 0. Pristine — like 173 (queue×prob) and 176 (color×prob), this is a clean synergy with no pre-existing entanglement.

**Available substrate beyond geometry/+optim/.**
- `linalg/decompose.go`: `LUDecompose`, `LUSolve`, `Inverse`, `Determinant`, `CholeskyDecompose`, `CholeskySolve` — needed for normal-equation least squares in SG1/SG2/SG5/SG13/SG15.
- `linalg/pca.go:33`: `PCA(data, nSamples, nFeatures, nComponents, components, explained)` — the Procrustes / Kabsch closed-form minimum-rotation step is structurally identical to the leading-singular-vector half of PCA; ready for SG5 reuse.
- `linalg/eigen.go:20`: `QRAlgorithm(A, n, eigenvalues, maxIter)` — eigenvalue-only (no eigenvectors), so SG14 Willmore (which needs the conformal Beltrami-Laplace eigenstructure) needs the eigenvector-returning peer flagged at 097-T1.
- `linalg/correlation.go:134`: `CovarianceMatrix(data, out)` — already covers the 3×3 SVD-input building block for Kabsch/Procrustes.
- `linalg/matrix.go`: `MatMul`, `MatTranspose`, `MatVecMul`, `Identity`, `MatAdd`, `MatScale`, `MatSub`, `Trace`, `CrossProduct` — the linalg API needed for SG5/SG6/SG13/SG15 is all present.
- `calculus/calculus.go`: `NumericalDerivative`, `NumericalGradient(f, x, h, out)`, `TrapezoidalRule`, `SimpsonsRule`, `GaussLegendre`, `MonteCarloIntegrate` — the gradient closure every optim entry-point demands is here when the caller cannot supply analytic; SG12 level-set evolution and SG11 snake also use Simpson's rule for arc-length integrals.
- `crypto/rng.go`: `MersenneTwister`, `PCG`, `Xoshiro256` — for SG7 (Lloyd Voronoi initialisation) and SG10 (Mitchell's-best-candidate sphere-pack initialisation).

**Pre-existing gaps blocking 4 of 20 primitives.**
- **(P1) Delaunay triangulation.** Repo grep `grep -ri 'Delaunay' --include='*.go'` → 0 matches. Required by SG7 Voronoi (Voronoi is the dual of Delaunay) and by SG14 Willmore (every meshed-surface curvature flow needs an underlying triangulation). Recommended placement: `geometry/triangulate.go`, ~280 LOC for Bowyer-Watson incremental construction in O(n log n) expected.
- **(P2) Half-edge mesh.** Zero matches in `geometry/` for `HalfEdge|Mesh|Vertex|Edge|Face` as a struct. Required by SG13 mesh geodesic, SG14 Willmore on meshes, SG17 conformal flattening. Recommended placement: `geometry/mesh.go`, ~180 LOC for the core `HalfEdgeMesh{Vertices, HalfEdges, Faces}` and `Vertex.OneRing()` traversal helpers.

Both gaps are independently flagged at slot 077-geometry-missing as Tier-1 — this synergy review reuses, never duplicates, those entries.

---

## 1. The twenty synergy primitives

Numbering SG1–SG20. For each: **(a) capability**, **(b) composition recipe** over present primitives, **(c) connective-tissue LOC**, **(d) blocking flag** if any.

### SG1 — `BezierCubicFit(points [][3]float64) (p0, p1, p2, p3 [3]float64)`

**(a)** Given a sequence of N≥4 sampled points (typically along a hand-drawn or noisy curve), find the four cubic-Bezier control points minimising sum-of-squared distances at uniform parameter spacing t_i = i/(N-1).

**(b)** Construct the 4-column Bernstein matrix B (N × 4 of B^3_j(t_i)), solve the normal equations BᵀB · P = Bᵀ X via `linalg.CholeskyDecompose` + `linalg.CholeskySolve` (BᵀB is 4×4 SPD by construction). Endpoint pinning P[0]=points[0], P[3]=points[N-1] reduces it to a 2×2 solve for the interior controls. For non-uniform parameter spacing, refine with one round of `optim.LBFGS` over (P1, P2) using `geometry.BezierCubic3D` as forward map and `calculus.NumericalGradient` for the gradient.

**(c)** ~80 LOC.

### SG2 — `SmoothingSplineGCV(xs, ys []float64) func(float64) float64`

**(a)** Cubic smoothing spline (Reinsch 1967) with the smoothing parameter λ chosen by leave-one-out generalised cross-validation (Craven-Wahba 1979). Trade-off between residual-sum-of-squares and integrated second-derivative roughness.

**(b)** Penalised least squares ‖y - Sλx‖² + λ‖S″x‖² has closed form per knot via the Reinsch banded system. Wrap the natural-cubic skeleton already in `optim/interpolate.go:CubicSplineNatural` (which solves a banded system for the second derivatives); add a smoothing term to the diagonal. GCV is `RSS(λ) / (1 - tr(H_λ)/N)²`; minimise it via `optim.GoldenSectionSearch` over `log10(λ) ∈ [-6, 2]`. ~120 LOC.

**(c)** 120 LOC.

### SG3 — `BSplineFit(points [][3]float64, degree, numCtrl int) []ControlPoint`

**(a)** Least-squares B-spline fit with prescribed degree (typically 3) and control-point count (much smaller than N). Used for surface fairing.

**(b)** Same normal-equation skeleton as SG1 with the cubic Bernstein basis swapped for the de-Boor recursion B^k_j(t). 90 LOC. (B-spline basis is not yet in `geometry/curves.go`; adding it here is ~30 LOC of de-Boor recursion + 60 LOC of normal-equation glue.)

**(c)** 90 LOC.

### SG4 — `BoundingBoxAlignedSDFFit(points [][3]float64) (center, halfExt [3]float64)`

**(a)** Given a point cloud, find the minimum-volume axis-aligned `SDFBox` enclosing it. The 6-D objective f(c, h) = `max_i SDFBox(points[i], c, h)` ≤ 0 with `Volume(h) = 8 * h[0]*h[1]*h[2]` minimised.

**(b)** Closed form for axis-aligned: c = 0.5*(min+max), h = 0.5*(max-min). ~20 LOC.

**(c)** 20 LOC. (Trivial standalone but a critical building block for SG10.)

### SG5 — `ProcrustesAlign(src, tgt [][3]float64) (R [9]float64, t [3]float64, s float64)`

**(a)** Rigid + uniform-scale alignment minimising ‖s·R·src + t - tgt‖² (Kabsch 1976; Umeyama 1991 for the scale). Closed-form via SVD of the 3×3 cross-covariance.

**(b)** Centroids via slice means; build 3×3 cross-covariance H = src_centeredᵀ · tgt_centered (use `linalg.MatMul` + `linalg.MatTranspose`). SVD H = UΣVᵀ — repo lacks 3×3 SVD today, but Kabsch needs only the optimal rotation R = V·D·Uᵀ where D = diag(1,1,sign(det(VUᵀ))) which is recoverable from `linalg.PCA` applied to the rotational matrix's symmetric part Hᵀ·H whose eigenvectors are V (eigenvalues are σ²). One QR pass on Hᵀ·H using `linalg.QRAlgorithm` recovers σ² but not eigenvectors — SG5 v1 ships a polar-decomposition iteration H ← 0.5*(H + H^{-T}) which converges quadratically to the orthogonal Procrustes solution and avoids the SVD-eigenvector gap. ~90 LOC.

**(c)** 90 LOC.

### SG6 — `ICP(src, tgt [][3]float64, maxIter int, tol float64) (R [9]float64, t [3]float64, finalRMSE float64)`

**(a)** Iterative Closest Point (Besl-McKay 1992 / Chen-Medioni 1991) for rigid alignment of two unordered point clouds. Outer loop: nearest-neighbour correspondence + SG5 Procrustes step until convergence.

**(b)** Outer loop = N-iterations of Procrustes; inner correspondence = brute-force O(N·M) nearest-neighbour search (kd-tree gating in v2). Convergence test on `finalRMSE` between successive iterations, tolerance 1e-6 default. ~150 LOC.

**(c)** 150 LOC. (90 LOC if SG5 already shipped.)

### SG7 — `LloydVoronoi(domain [4]float64, n int, maxIter int) [][2]float64`

**(a)** Centroidal Voronoi tessellation (Lloyd 1982): n sites in 2D AABB `domain`, alternately recomputing each Voronoi cell's centroid and snapping the site to it. Used for blue-noise sampling, mesh generation, k-means.

**(b)** Per-iteration: compute Voronoi diagram (dual of Delaunay triangulation = **P1 gap**). Approximate via the Jump-Flood algorithm on a grid (no Delaunay required) for v1 — ~140 LOC. For exact O(n log n) construction, gate on P1.

**(c)** 140 LOC for grid-approximate; 250 LOC for exact-Delaunay version (gated on P1).

**(d)** Blocked on P1 for the exact path; the grid-Jump-Flood path ships standalone.

### SG8 — `EarthMoverDistance2D(x, y [][2]float64) float64`

**(a)** 2D earth-mover / Wasserstein-2 distance between two empirical 2D point distributions (Sinkhorn-regularised in v1).

**(b)** Build N×N pairwise cost matrix C[i][j] = ‖x_i - y_j‖² using `linalg.L2Norm`-equivalent inline math. Call `optim/transport.Sinkhorn(C, μ=1/N, ν=1/N, ε=0.05, ...)` directly — the existing log-domain Sinkhorn IS the 2D EMD when fed a Euclidean cost matrix. ~50 LOC. Pure composition.

**(c)** 50 LOC.

### SG9 — `BoundedConvexHull(points [][2]float64, maxArea float64) [][2]float64`

**(a)** Constrained convex hull: find the convex hull whose area does not exceed `maxArea` by snipping minimum-perimeter-loss triangles off the original Graham hull.

**(b)** Compute `geometry.ConvexHull2D(points)`, compute area via `geometry.TriangleArea2D` triangle-fan summation. Greedily remove the vertex contributing the largest triangle (sorted by `TriangleArea2D`) until area ≤ maxArea. ~80 LOC. Pure composition.

**(c)** 80 LOC.

### SG10 — `SpherePackRandom(domainCenter, domainHalfExt [3]float64, radius float64, n int, rng) [][3]float64`

**(a)** Mitchell's-best-candidate sphere packing: sample n point-centers in `SDFBox(domainCenter, halfExt)`, each maximising the minimum SDF value to all previously-placed spheres.

**(b)** Inner loop: M=20 candidate samples, evaluate `min over placed: dist - radius` via `geometry.SDFSphere` per candidate, keep the best. Caller supplies `crypto.Xoshiro256.Float64()`. ~110 LOC. Pure composition over `SDFBox` (containment) + `SDFSphere` (per-pair distance).

**(c)** 110 LOC.

### SG11 — `SnakeActiveContour(image [][]float64, x0 [][2]float64, alpha, beta, gamma float64, maxIter int) [][2]float64`

**(a)** Kass-Witkin-Terzopoulos 1988 snake: minimise E = α·E_continuity + β·E_curvature + γ·E_image-gradient over a closed parametric curve. Gradient flow recovers a contour locked to image edges.

**(b)** The Euler-Lagrange tridiagonal system A·X^{t+1} = X^t + γ·∇E_image solves once at construction (A is α/β-only constant tridiagonal banded matrix, factorised by `linalg.LUDecompose` in O(N)). Per iteration: backward-Euler step with image-gradient force. Image-gradient force can ride on `signal.Convolve` 2D Sobel via `signal/2d` (currently absent — caller-supplied gradient field for v1). ~180 LOC + 60 LOC of test.

**(c)** 180 LOC.

### SG12 — `LevelSetEvolve(phi [][]float64, speed func([2]float64) float64, dt float64, steps int) [][]float64`

**(a)** Osher-Sethian 1988 level-set method: evolve the implicit surface φ(x) = 0 under a normal-direction speed field via the Hamilton-Jacobi PDE φ_t + F·‖∇φ‖ = 0, integrated by upwind first-order finite differences with reinitialisation every K steps to keep φ a valid signed-distance function.

**(b)** Per-step upwind discretisation uses Godunov's scheme on the |∇φ| term (50 LOC). Reinitialisation step: solve eikonal ‖∇φ‖ = 1 in steady state via `optim.GoldenSectionSearch` per voxel (or fast-marching, deferred). Speed callback can wrap `geometry.SDFSphere` etc. for analytic test cases. ~200 LOC + 80 LOC of test.

**(c)** 200 LOC. CROWN JEWEL with SG11.

### SG13 — `MeshGeodesic(mesh HalfEdgeMesh, src, dst int) []int`

**(a)** Mitchell-Mount-Papadimitriou 1987 exact polyhedral geodesic on triangle mesh. Returns the path through faces and edges from source to destination vertex.

**(b)** Window propagation across edges (windows = isoparametric arc segments); priority-queue advancement using `graph.DijkstraVertexPredecessor` skeleton on the half-edge graph (graph package's heap-PQ already shipped per slot 124-Q11). ~280 LOC.

**(c)** 280 LOC.

**(d)** Blocked on P2 (half-edge mesh).

### SG14 — `WillmoreFlow(mesh HalfEdgeMesh, dt float64, steps int) HalfEdgeMesh`

**(a)** Mean-curvature gradient flow on the Willmore energy W(M) = ∫(H² - K) dA, where H is mean curvature, K is Gaussian curvature. Drives surfaces toward the sphere — the unique Willmore minimiser.

**(b)** Per-vertex compute discrete H via cotangent Laplacian (Pinkall-Polthier 1993) on one-ring; gradient step v ← v - dt · ΔH·n via `optim.GradientDescent` with externally-supplied (analytic) gradient closure. ~220 LOC + 80 LOC of test against a sphere golden file.

**(c)** 220 LOC.

**(d)** Blocked on P2 (half-edge mesh) + needs `linalg.SolveCotangentLaplacian` (sparse positive-definite ~120 LOC, flagged at 097-T1).

### SG15 — `GeneralizedProcrustes(shapes [][][3]float64, maxIter int) (mean [][3]float64, aligned [][][3]float64)`

**(a)** Kendall 1984 generalised Procrustes analysis on a population of K shapes (each a length-N point cloud). Iteratively estimate consensus mean and align all K shapes to it. Foundation of statistical shape analysis.

**(b)** Outer loop: align each shape_i to current mean via SG5 Procrustes; recompute mean = elementwise-average over K aligned shapes; convergence test on mean-shift Frobenius norm. ~80 LOC after SG5 ships. Pure composition.

**(c)** 80 LOC.

### SG16 — `BezierLengthMinimizer(p0, p3 [3]float64, knots [][3]float64) (p1, p2 [3]float64)`

**(a)** Given pinned endpoints p0, p3 and a sequence of intermediate "must-pass" knots, find interior controls (p1, p2) minimising arc length subject to the knot-passing constraint at uniform t.

**(b)** Arc length L = ∫₀¹ ‖B'(t)‖ dt evaluated by `calculus.GaussLegendre(speed, 0, 1, 8)` where speed is the analytic Bezier derivative magnitude. Constrained optimisation via `optim/proximal.Fbs` with quadratic-penalty for knot-passing or via Lagrange-multiplier reduction. ~140 LOC.

**(c)** 140 LOC.

### SG17 — `DiscreteConformalFlatten(mesh HalfEdgeMesh) [][2]float64`

**(a)** Pinkall-Polthier 1993 discrete conformal map: flatten a topological-disk mesh to the plane preserving angles up to scale. Lays out a 3D mesh in 2D for texturing or analysis.

**(b)** Cotangent-Laplacian linear system `L·u = b` where `b` encodes boundary constraints (one boundary vertex pinned, one boundary direction fixed). Solve via `linalg.LUSolve` after sparse-Cholesky-equivalent dense factorisation for v1. ~180 LOC.

**(c)** 180 LOC.

**(d)** Blocked on P2 (half-edge mesh).

### SG18 — `InverseKinematics(skeleton []Joint, target [3]float64, maxIter int) []float64`

**(a)** Compute joint angles θ ∈ R^k such that forward-kinematics FK(θ) = target. Constrained nonlinear optimisation.

**(b)** Forward-kinematics is `geometry.QuatFromAxisAngle` × `geometry.QuatRotateVec` chain along skeleton. Optimise ‖FK(θ) - target‖² via `optim.LBFGS` with `calculus.NumericalGradient`-supplied Jacobian. Optionally use `optim/proximal.ProxBox(jointLimits)` to enforce angle limits via Fbs. ~160 LOC.

**(c)** 160 LOC. (90 LOC if `autodiff.Tape` ships per slot 014; gradient becomes free.)

### SG19 — `MinimumVolumeBoundingSphere(points [][3]float64) (center [3]float64, radius float64)`

**(a)** Welzl 1991 minimum enclosing ball / smallest-enclosing-sphere of a point cloud.

**(b)** Welzl's randomised incremental algorithm with `crypto.Xoshiro256` for shuffling. Base cases (sphere through 0/1/2/3/4 points) are closed-form. Falls through `geometry.SDFSphere` evaluations for containment tests. ~140 LOC.

**(c)** 140 LOC.

### SG20 — `ShapeDerivativeFiniteDifference(omega SDF, J func(SDF) float64, h float64) func([3]float64) float64`

**(a)** Numerical shape derivative dJ/dΩ via Hadamard's structure theorem: dJ/dΩ in the direction θ equals ∫_∂Ω g(x) · θ(x)·n(x) ds for some scalar shape gradient g. Approximate g pointwise by perturbing the SDF level-set and observing J change.

**(b)** Caller supplies `omega` as an `SDF func([3]float64) float64` closure (e.g. `func(p [3]float64) float64 { return geometry.SDFSphere(p, [3]float64{0,0,0}, 1.0) }`) and J as a functional. Perturb at point p by replacing omega(x) with omega(x) - h·δ_p(x) (smoothed indicator) and finite-difference J. ~110 LOC.

**(c)** 110 LOC. Building block for Hadamard-formula shape optimisation; couples cleanly with SG12 to drive level-set evolution by an arbitrary shape functional.

---

## 2. Composition matrix

| Primitive | geom uses | optim uses | linalg uses | calculus uses | crypto uses | other |
|---|---|---|---|---|---|---|
| SG1 BezierFit | BezierCubic3D | LBFGS (refine) | Cholesky | NumGrad | — | — |
| SG2 SplineGCV | — | GoldenSection, CubicSplineNatural | banded solve | — | — | — |
| SG3 BSplineFit | (de-Boor adds to curves.go) | LBFGS | Cholesky | NumGrad | — | — |
| SG4 BoxFit | SDFBox | — | — | — | — | — |
| SG5 Procrustes | — | — | MatMul, MatTranspose | — | — | (SVD via polar iter) |
| SG6 ICP | — | — | (via SG5) | — | — | — |
| SG7 Lloyd | — | — | — | — | Xoshiro256 | (P1 Delaunay v2) |
| SG8 EMD2D | — | Sinkhorn | L2Norm | — | — | — |
| SG9 BoundedHull | ConvexHull2D, TriangleArea2D | — | — | — | — | — |
| SG10 SpherePack | SDFSphere, SDFBox | — | — | — | Xoshiro256 | — |
| SG11 Snake | — | — | LU | NumGrad | — | (signal.Convolve2D) |
| SG12 LevelSet | (SDF closure) | GoldenSection | — | — | — | — |
| SG13 Geodesic | (P2 mesh) | — | — | — | — | graph.Dijkstra |
| SG14 Willmore | (P2 mesh) | GradDescent | (sparse Lap) | — | — | — |
| SG15 GenProcrustes | — | — | (via SG5) | — | — | — |
| SG16 BezierLen | BezierCubic3D | proximal.Fbs | — | GaussLegendre | — | — |
| SG17 ConfFlatten | (P2 mesh) | — | LUSolve | — | — | — |
| SG18 IK | QuatFromAxisAngle, QuatRotateVec | LBFGS, ProxBox | — | NumGrad | — | — |
| SG19 MinSphere | SDFSphere | — | — | — | Xoshiro256 | — |
| SG20 ShapeDeriv | (any SDF closure) | — | — | (smoothed δ) | — | — |

Verified composition surface — no SG primitive needs more than the named existing entries plus P1/P2.

---

## 3. Connective-tissue LOC totals & PR sequencing

- **PR-1 ship-today fundamentals:** SG1+SG2+SG9 = **310 LOC** (Bezier fit + smoothing-spline GCV + bounded hull). One-day PR. Saturates R-MUTUAL-CROSS-VALIDATION 3/3 pin: de-Casteljau-deterministic forward-eval × LBFGS-numerical fit × normal-equation-closed-form on randomly-generated point clouds, agreeing to 1e-10 (mirroring 6a55bb4 audio-onset-3-detector and 365368a Clayton-autodiff-vs-analytic).

- **PR-2 registration toolkit:** SG5+SG6+SG15+SG19 = **460 LOC** (Procrustes, ICP, Generalized Procrustes, MinBoundingSphere). Two-day PR. Highest-leverage single block — SLAM/RubberDuck-3D/point-cloud-registration consumers all want this and zero-dep-Go ships nothing today. Saturates R-MUTUAL pin (Kabsch SVD route × polar-iteration route × ICP-fixpoint route on rigid-rotation golden vectors).

- **PR-3 SDF/sphere-pack/spline:** SG3+SG4+SG10+SG18 = **380 LOC** (BSpline fit, axis-aligned box fit, sphere-pack, IK). Two-day PR. SG18 is the single biggest cross-link to robotics consumers.

- **PR-4 transport+activecontour:** SG8+SG11 = **230 LOC** (EMD2D + Snake). One-day PR.

- **PR-5 level-set crown jewel:** SG12+SG16+SG20 = **450 LOC** (Hamilton-Jacobi level-set evolution + Bezier-length minimisation + finite-difference shape derivative). Two-day PR. Saturates second R-MUTUAL pin (level-set evolution from sphere-IC under constant negative speed × SDFSphere shrinking analytic × eikonal reinit golden-file all agree to 1e-6 per step).

- **PR-6 Voronoi grid path:** SG7 (jump-flood variant) = **140 LOC**. Half-day PR.

- **PR-7 mesh family** (gated on P1 + P2 substrate): SG7-exact + SG13 + SG14 + SG17 = **930 LOC** + ~460 LOC P1+P2 substrate = **1,390 LOC**. ~2 weeks. Crown jewel — turns reality into a competent computational-geometry / geometry-processing library.

**Totals.** PR1–PR6: **1,970 LOC ship-today** in ~1 sprint (~7 engineer-days). PR7: **+1,390 LOC** gated on P1 + P2 substrate from slot 077 (3-week extra). Combined: **3,360 LOC source + ~1,500 LOC tests** for full-spectrum coverage.

---

## 4. Recommended placement

Three new sub-packages, cycle-free DAG, mirroring 158/161/165/170/171/172/173/174/176 sixteen-consecutive-synergy consumer-side-placement convention:

```
geometry/fit/         (PR-1 + PR-3)
   bezier_fit.go      SG1, SG3, SG16
   spline_gcv.go      SG2
   sdf_fit.go         SG4
   sphere_pack.go     SG10, SG19
   ik.go              SG18
   convex_constrained.go  SG9
   shape_deriv.go     SG20
   imports: geometry/, optim/, optim/proximal, linalg/, calculus/, crypto/

geometry/registration/ (PR-2)
   procrustes.go      SG5
   icp.go             SG6
   generalized.go     SG15
   imports: geometry/, optim/, linalg/

geometry/levelset/    (PR-4 + PR-5)
   snake.go           SG11
   evolve.go          SG12
   transport2d.go     SG8 (rehomed from optim/transport per consumer-side convention)
   imports: geometry/, optim/, optim/transport, calculus/

geometry/voronoi/     (PR-6, exact path = PR-7 gated)
   lloyd.go           SG7
   imports: geometry/, optim/, crypto/

geometry/mesh/        (PR-7 gated)
   half_edge.go       P2 substrate
   delaunay.go        P1 substrate
   geodesic.go        SG13
   willmore.go        SG14
   conformal.go       SG17
   imports: geometry/, optim/, linalg/, graph/
```

Cycle-free DAG: every new sub-package depends on `geometry/` + `optim/` (+ siblings); reverse-direction edges never. Zero new abstraction needed (no new interface types) — every primitive composes existing function signatures.

---

## 5. Precision hazards & golden-file pinning

Documented for caller-side and cross-language validation:

- **SG1 BezierFit at-least-4-points** invariant; `len(points) < 4` returns ErrInsufficientPoints (matches NIST cubic-spline conventions).
- **SG2 SmoothingSpline log-λ search** must bracket [-6, +2] not [-12, +12] — outside this the Reinsch system becomes ill-conditioned and GCV is non-monotone.
- **SG5 Kabsch sign-correction** `D = diag(1, 1, sign(det(VUᵀ)))` — without this, an improper rotation (reflection) is returned for left-handed point clouds. Pin via golden file containing a reflected point pair.
- **SG6 ICP local-minima** documented: ICP converges to nearest local minimum, not global. Recommend caller seed via PCA-axis alignment (SG5 with no rotation) before iterating.
- **SG7 Lloyd convergence** monotone in CVT energy by Du-Faber-Gunzburger 1999 but slow near the minimum; cap at 50 iterations and warn-not-fail on non-convergence.
- **SG10 Mitchell-best-candidate non-uniqueness** — the candidate count M=20 is the de-facto standard but produces different placements per RNG seed; pin seed in golden file.
- **SG11 Snake α/β range** — α ∈ [0.001, 0.1] for elasticity, β ∈ [0.001, 0.1] for curvature, γ ∈ [0.1, 5] for edge-attraction. Outside these, the contour either freezes or explodes; document explicitly.
- **SG12 LevelSet CFL condition** dt ≤ Δx / max|F|; violating this produces upwind-scheme oscillation. Validate at construction; return ErrCFLViolation rather than silently produce garbage.
- **SG12 reinitialisation cadence** every 5–10 steps; always before extracting the zero level set.
- **SG13 MMP geodesic exact-arithmetic note** — the published algorithm assumes exact predicates; Float64 rounding can flip window dominance. Robust-predicates substrate (Shewchuk 1997) is independently flagged at slot 077; gate exact-MMP on it.
- **SG14 Willmore time step** scales with edge-length-squared; recommend dt = 0.1 · min_edge². Larger steps blow up.
- **SG18 IK rank-deficient Jacobian** at gimbal-lock or extended-arm config — Levenberg-Marquardt damping (already inside LBFGS as Hessian regularisation) covers most cases; recommend documenting "may not converge near singular configurations".
- **SG19 Welzl move-to-front heuristic** — random shuffling of `points` is what gives O(n) expected time; using sorted input regresses to O(n²). Pin seed in golden file.
- **SG20 finite-difference h** scales with the SDF length scale; document recommended h = 0.01 · diam(Ω) and warn that smooth indicator δ_p has to integrate to 1 in the discretisation, not just be positive.

Cross-language byte-determinism: CGAL (C++), scipy.spatial (Python), Statistics.jl (Julia) all ship reference Procrustes/ICP/spline-fit; pin SG1/SG2/SG5/SG6/SG10 against scipy.optimize / scipy.interpolate / scipy.spatial golden vectors at 1e-9 absolute tolerance per CLAUDE.md sec1 conventions.

---

## 6. Why this synergy is high-leverage

1. **Twenty primitives, sixteen ship today, no new abstraction.** Of the 20, only SG7-exact / SG13 / SG14 / SG17 are blocked, all on the same two missing-substrate gaps (P1 Delaunay + P2 half-edge mesh) which slot 077 already names — this review reuses, never duplicates.
2. **The optimisation surface is mostly already there.** `optim.LBFGS` + `optim.GoldenSectionSearch` + `optim/proximal.ProxBox` + `optim/transport.Sinkhorn` cover nineteen of twenty primitive-needs. No new optimiser required.
3. **Closed-form composition wherever possible.** SG1/SG4/SG5/SG8/SG9/SG15/SG19 all reduce to closed-form linear-algebra solves over present `linalg/` primitives. Iterative routes (LBFGS, golden-section, Sinkhorn) are correctness backups, not the headline path — exactly the design philosophy CLAUDE.md sec1 ("golden files are the proof") rewards.
4. **Three R-MUTUAL-CROSS-VALIDATION pins fall out for free:** Bezier-fit-3-way (PR-1), Procrustes-3-way (PR-2), Level-set-3-way (PR-5), each saturating the same R-pattern as commits 6a55bb4 (audio-onset 3-detector) and 365368a (Clayton autodiff-vs-analytic).
5. **Pristine zero-cross-edge starting point.** Like 173 (queue×prob) and 176 (color×prob), this synergy has zero pre-existing entanglement — the cleanest possible composition surface.
6. **Seventeenth consecutive consumer-side-placement synergy.** 158/159/160/161/165/166/167/168/169/170/171/172/173/174/175/176 + this 177 = 17 successive synergies all placing the new code in NEW sub-packages of the consumer side, never modifying the supplier; this convention is now beyond doubt.

---

## 7. Cross-references to sibling reviews

- **076-080 (geometry isolation)**: 077-T1 names half-edge-mesh and Delaunay as in-package gaps (P1 + P2 here). 078-sota names CGAL/scipy.spatial as cross-language pinning targets. 080-perf names allocation-free SDF evaluation as Pistachio-60-FPS hot path — every SG primitive that calls SDF in its inner loop (SG10/SG12/SG19/SG20) inherits this.
- **101-105 (optim isolation)**: 102-T1 enumerates Adam/RMSprop/SGD-Nesterov as missing modern optimisers (orthogonal — this synergy does not need them); 102-T2.21 names BO/EI for hyperparam search (orthogonal); 104-api flags the `func(x []float64, out []float64)` gradient convention which every SG primitive consuming optim adheres to (SG1/SG3/SG6/SG11/SG14/SG18).
- **097 (linalg-missing T1)** flags eigenvector-returning SVD/QR — gates SG14 Willmore via cotangent-Laplacian eigendecomposition.
- **163 synergy-optim-autodiff**: independent first-cousin synergy ships forward-mode duals and reverse-mode tape; if PR-2 of 163 lands first, SG18 IK gradient becomes free (no `calculus.NumericalGradient`) — pure decoration, not necessity.
- **174 synergy-gametheory-optim**: shares the optim/proximal substrate — `ProxBox` (SG18 joint limits), `ProxSimplex` (gametheory G3) — orthogonal axes consuming the same prox library.
- **164 synergy-orbital-optim**: shares the LBFGS-for-trajectory-fitting pattern; SG18 IK and orbital-trajectory-shooting are structurally identical (both are "find parameters of a kinematic chain such that endpoint matches target").
- **168 synergy-physics-autodiff**: shape-derivative SG20 is the geometry analogue of physics-Lagrangian-derivative — both compose the closure-of-functional-with-perturbation idiom.
- **176 synergy-color-prob**: orthogonal axis (random-color-and-uncertain-color vs shape-fitting-and-shape-evolution) but shares the Halton/Sobol QMC primitive (CP3 there) which would refine SG10 Mitchell-best-candidate from random to QMC-deterministic — pure decoration cross-link only.

This synergy is **first geometry × optim review in the 400-sequence**. Distinct from every prior slot.

---

## 8. Single-day high-leverage commit recommendation

If you can ship **one** PR from this review, ship **PR-2 registration toolkit** (SG5+SG6+SG15+SG19 = 460 LOC, two engineer-days). Reasons:

- **Largest external-consumer demand:** Procrustes/ICP is the foundational primitive of every SLAM, point-cloud-registration, motion-capture, structure-from-motion, and computational anatomy pipeline. Zero zero-dep-Go library ships these today.
- **Zero new abstraction:** all four primitives compose existing `linalg/` (MatMul/MatTranspose/Cholesky) + existing `geometry/` (SDFSphere for MinSphere containment).
- **Saturates an R-MUTUAL pin:** Kabsch-SVD-route × polar-iteration-route × ICP-fixpoint-route on a rigid-rotation golden file from scipy.spatial.transform.Rotation.
- **Single biggest cross-link to the named consumer surface:** RubberDuck-3D, robotics, SLAM all cite "do we have ICP?" as the first question.

Recommended PR-1 if a half-day budget: SG9 BoundedConvexHull = 80 LOC. Self-contained, pure composition over `geometry.ConvexHull2D` + `geometry.TriangleArea2D`, no new sub-package required (lands in `geometry/polygon.go` directly).

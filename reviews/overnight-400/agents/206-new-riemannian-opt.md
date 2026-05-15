# 206 | new-riemannian-opt

**Summary (2 lines):** reality v0.10.0 ships ZERO Riemannian-optimisation surface — `optim/` exposes only Euclidean GD/L-BFGS/simplex/IP/GA/SA/rootfind/interpolate plus sub-packages `optim/proximal/` (FBS/FISTA/ADMM with Euclidean prox-projections only) and `optim/transport/` (1-D Wasserstein + Sinkhorn, no manifold structure); `geometry/` ships `quaternion.go` (S³ ≅ SU(2), the *only* manifold-as-such anywhere in the repo, with `QuatSlerp` as the *only* shipped retraction-equivalent) plus SDF/curves/polygon; `infogeo/` ships f-divergences/Bregman/MMD only and explicitly defers Fisher-Rao/SPD/Stiefel/Grassmann/hyperbolic to v2 per `092-infogeo-missing.md` Tier-2; repo-wide grep on `Manifold|Stiefel|Grassmann|Retract|ParallelTransport|Riemann(?!-friendly)` returns zero callable surface (only doc-string mentions in 092/177 reviews + one comment in `prob/copula/gaussian.go:217` "Riemann-friendly on a bounded interval" referring to Riemann *integration*, not Riemannian *geometry*). This slot scopes the Riemannian-optimisation canon as twenty-two primitives R1–R22 totalling ~3,200 LOC with `optim/manifold/` as a new sub-package (mirroring the `optim/proximal/` + `optim/transport/` precedent) holding the `Manifold` / `RiemannianMetric` interface + retraction-based optimisers + concrete manifolds (Sphere/Stiefel/Grassmann/SPD/Hyperbolic), with **the Sphere manifold + Riemannian-GD + QR-retraction-on-Stiefel triple (~480 LOC) as the Tier-1 keystone** because it validates the entire `Manifold` contract on the simplest closed-form case and unblocks every downstream consumer (PCA-on-Stiefel, online subspace tracking, Procrustes alignment, hyperbolic embeddings). Disambiguation versus 092-infogeo-missing-T2.1–T2.7 (`Manifold` interface + Sphere/Hyperbolic/SPD/Stiefel/Grassmann *as statistical manifolds*): 092 scoped manifolds in *information-geometry* terms (Fisher-Rao on simplex, SPD-as-Gaussian-Σ, e/m geodesics); this slot scopes the same manifolds as *optimisation domains* (argmin f(x) s.t. x ∈ M with retractions and vector transport) — overlap is intentional and the recommended placement is shared (`optim/manifold/sphere.go` ships Sphere primitives that 092 T2.4 imports for Fisher-Rao geodesics). Disambiguation versus 205-new-lie-groups-L15 (Lie-Newton / Lie-GD / ESM on SO(3)/SE(3), 250 LOC in `optim/manifold.go`): 205 scoped *Lie-group-specific* retraction optimisation; this slot is the *generic-manifold* superset (Stiefel, Grassmann, SPD, hyperbolic — none of which are Lie groups). Same `optim/manifold/` placement; 205-L15 ships SO(3)/SE(3) as concrete `Manifold` instances; this slot ships the interface plus the non-Lie manifolds.

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Verified by direct read.

### `optim/` (top-level, ~1,800 LOC across 11 source files)
- `gradient.go` (250 LOC): `GradientDescent(f, grad, x0, lr, maxIter, tol)` and `LBFGS(f, grad, x0, m, maxIter, tol)`. Both take Euclidean gradients (`grad: func([]float64, []float64)` writes ∇f into a slice); both update `x ← x − step·d` with no projection step; both fall over silently if the caller's `x` lives on a manifold (the iterates drift off without complaint).
- `gradient_validated.go` (221 LOC): R123 input-check wrappers around the above. Same Euclidean-only update rule.
- `linear.go` (286 LOC): SimplexMethod-Bland + InteriorPoint-barrier-gradient. LP-only.
- `genetic.go`, `metaheuristic.go`, `rootfind.go`, `interpolate.go`: black-box / 1-D / interp; no manifold structure.
- `proximal/` (FBS/FISTA/ADMM + 8 prox ops): `ProxL2Ball` is the *only* shipped operator that projects onto a curved set, and it is the projection onto the unit Euclidean ball (an indicator-function-based prox), not a retraction onto a Riemannian manifold. `ProxSimplex` projects onto Δⁿ⁻¹ (a flat polytope), not Fisher-Rao simplex.
- `transport/`: 1-D Wasserstein + Sinkhorn. The Sinkhorn iteration is *implicitly* a Bregman-projection on the joint-distribution simplex but neither the Bregman geometry nor a `Manifold` abstraction is exposed.

### `geometry/` (~720 LOC)
- `quaternion.go` (266 LOC): the *only* manifold operations in the entire repo. `QuatSlerp` is structurally a retraction on S³ (the `t·log(R0⁻¹·R1)` path lifted by `R0·exp(·)`); `QuatFromAxisAngle` / `QuatToAxisAngle` are exp/log on S³ ≅ SU(2). None of these are *labelled* as manifold operations — the API is geometric-rotation-by-rotation, not manifold-by-manifold. Per slot 205-new-lie-groups, the SU(2) callable surface needs aliasing (40 LOC) to make the manifold identity discoverable.
- `sdf.go`, `curves.go`, `polygon.go`: zero manifold surface.

### `infogeo/` (~1,373 LOC across 4 files)
- f-divergences (KL/JS/TV/Hellinger/χ²/Rényi-α), Bregman (squared-Euclidean/genKL/IS/generic), MMD² with RBF kernel. *Zero* Fisher-Rao, *zero* α-connections, *zero* dual-flat structure, *zero* manifold operations — explicitly per `infogeo/doc.go:55-79` MVP scope. 092-infogeo-missing T2.1–T2.7 enumerates the gap.

### `chaos/` (~600 LOC)
- `RK4Step`, `EulerStep`, `SolveODE`. No symplectic integrator; therefore no manifold-preserving geodesic flow (per slot 091 §2.2 + slot 204). `chaos.Verlet` is the cross-package blocker for any manifold ODE.

### `linalg/`
- `MatMul`, `LUDecompose`, `QRDecompose`, `Cholesky`, `Eigen` (Jacobi for symmetric, eigenvalues-only for general). `MatrixExp` and `MatrixLog` *not* shipped (per 097-T1, 092-blocker, 205-§0). SVD eigenvector-returning peer not shipped (per 097-T1). Both are sub-blockers for SPD-manifold operations (T2.3 in 092 spec).

### Repo-wide grep audit
```
$ grep -rn "Manifold\|Stiefel\|Grassmann\|Retract\|ParallelTransport\|VectorTransport\|GeodesicShoot" --include="*.go"
(zero hits in any source file outside reviews/)
```
```
$ grep -rn "Riemann" --include="*.go"
infogeo/doc.go:7        # docstring mention "statistical manifolds"
prob/copula/gaussian.go:217   # "Riemann-friendly on a bounded interval" (Riemann integration)
```
Two doc-string mentions, *zero* callable surface. The single closest-to-manifold-optim primitive in the repo is `proximal.ProxSimplex` which does Euclidean projection onto Δⁿ⁻¹ — useful as a building block but not a Riemannian retraction (it doesn't respect the Fisher-Rao metric on the interior).

### Cross-link audit
- **092-infogeo-missing T2.1–T2.7**: scoped Sphere/Hyperbolic/SPD/Stiefel/Grassmann as statistical manifolds *for IG purposes* (Fisher-Rao geodesics, e/m projections, natural-gradient). This slot scopes the *same manifolds* as optimisation domains. Recommended overlap: `optim/manifold/sphere.go` ships the Sphere ONCE; `infogeo/manifold_fisher.go` decorates it with the Fisher-Rao metric. No duplication.
- **205-new-lie-groups L15**: scoped Lie-Newton / Lie-GD / ESM on SO(3)/SE(3) (250 LOC in `optim/manifold.go`). This slot's `optim/manifold/` package is the destination — 205-L15 ships SO(3) and SE(3) as concrete `Manifold` implementations against this slot's interface.
- **177-synergy-geometry-optim**: scoped 20 synergy primitives across `geometry/fit/` + `geometry/levelset/` + `geometry/registration/`. SG5 (Procrustes alignment) and SG6 (ICP) are structurally Riemannian-optimisation problems (orthogonal-Procrustes lives on Stiefel V(n,n) = O(n)); recommended re-expression once R-package ships.
- **102-optim-missing**: enumerates 50 Euclidean-optim gaps. *Riemannian optimisation is not on the list* (102 explicitly stays in Euclidean territory). This slot fills the gap that 102 declined to scope.
- **204-new-symplectic-int**: scoped `chaos/symplectic.go` (Verlet/leapfrog/Yoshida ~140 LOC keystone). Required cross-package blocker for R10 (geodesic ODE solver) and R14 (RMHMC).

---

## 1. The twenty-two primitives

Numbered R1–R22. For each: **(a) what reality ships**, **(b) what to add**, **(c) connective LOC**, **(d) blocker if any**.

### R1 — `Manifold` / `RiemannianMetric` interface

(a) **Ships:** nothing. The closest analogue is `prob.Distribution interface { PDF, CDF }` — domain-specific, no manifold structure.

(b) **Add:** the Geomstats / Pymanopt / Manopt.jl-shaped contract:
```go
type Manifold interface {
    Dim() int
    AmbientDim() int
    Project(x, out []float64) error                       // closest point on M (retraction-projection)
    InnerProduct(x, u, v []float64) float64               // Riemannian metric ⟨u,v⟩_x
    Norm(x, u []float64) float64                          // = sqrt(InnerProduct(x, u, u))
    RandomPoint(rng Rand, out []float64) error            // uniform-on-M sampling
    RandomTangent(x []float64, rng Rand, out []float64) error
}

type RiemannianMetric interface {
    Manifold
    Exp(x, v, out []float64) error                        // geodesic from x with initial velocity v
    Log(x, y, out []float64) error                        // initial velocity of geodesic from x to y
    Retract(x, v, out []float64) error                    // 1st-order approx of Exp (cheaper)
    InverseRetract(x, y, out []float64) error             // 1st-order approx of Log
    GeodesicDistance(x, y []float64) float64
    ProjectTangent(x, v, out []float64) error             // P_x: T_x_ambient → T_x M
    EuclideanToRiemannianGradient(x, gradEucl, out []float64) error  // ∇^M f = P_x(∇f)
    ParallelTransport(x, y, v, out []float64) error       // closed-form when available
    VectorTransport(x, y, v, out []float64) error         // 1st-order approx of PT (cheaper)
}
```
Reference: Absil-Mahony-Sepulchre 2008 *Optimization Algorithms on Matrix Manifolds* (AMS) §3.6 (retractions), §8 (vector transport); Boumal 2023 *An Introduction to Optimization on Smooth Manifolds* §3.4–§3.6, §10.3.

(c) **LOC:** ~120 in `optim/manifold/manifold.go`. Interface + `BaseManifold` mixin (zero-method default impl plumbing) + `RandomTangent` adapter that Gram-Schmidts a Gaussian sample.

(d) **No blocker.**

### R2 — Sphere manifold S^{n-1}

(a) **Ships:** S³ ⊂ ℝ⁴ implicit in `geometry/quaternion.go` (`QuatSlerp`, `QuatFromAxisAngle`, `QuatToAxisAngle`, `QuatNormalize`). Not abstracted as a generic S^{n-1}; not implementing any `Manifold` interface (which doesn't exist yet).

(b) **Add:** the textbook closed-form everything for `S^{n-1} = {x ∈ ℝⁿ : ‖x‖ = 1}`:
- Project: `x ← x/‖x‖`.
- InnerProduct: standard ℝⁿ dot (since `T_x S^{n-1}` is a linear subspace).
- Exp: `Exp_x(v) = cos(‖v‖)·x + sin(‖v‖)/‖v‖·v` with `‖v‖ → 0` Taylor fallback (Geomstats §4.2).
- Log: `Log_x(y) = (θ/sin θ)·(y − cos(θ)·x)` where `θ = arccos(⟨x,y⟩)` clamped to `[−1, 1]`; antipodal `θ = π` direction-ambiguous (document with arbitrary tangent direction selection).
- Retract: `Retract_x(v) = (x + v)/‖x + v‖` — the *projection retraction*, 1st-order accurate, ~3× cheaper than Exp.
- ProjectTangent: `P_x(v) = v − ⟨x, v⟩·x` (orthogonal projection onto `T_x S^{n-1}`).
- ParallelTransport closed-form: `PT_{x→y}(v) = v − (⟨log x y, v⟩ / ‖log x y‖²)·(log_x y + log_y x)` (Edelman-Arias-Smith 1998, equivalent to the Schild's-ladder closed-form for spheres).
- GeodesicDistance: `arccos(⟨x, y⟩)`.

Reference: AMS §3.6.1 (sphere retraction), §8.1.2 (sphere PT); Edelman-Arias-Smith 1998 *SIMAX* 20:303 §2.2; Boumal 2023 §3.6.1.

(c) **LOC:** ~140 in `optim/manifold/sphere.go`. Pairs with 092 T2.4 (Fisher-Rao simplex via the `√p` sphere embedding) — `infogeo/manifold_fisher.go` should *consume* this file rather than ship its own sphere code.

(d) **No blocker.** Cleanest manifold to ship first; validates the entire interface.

### R3 — Stiefel manifold St(n, p) with QR retraction and polar retraction

(a) **Ships:** nothing. `linalg.QRDecompose` exists (~150 LOC, modified-Gram-Schmidt) — needed for the QR retraction.

(b) **Add:** `St(n, p) = {X ∈ ℝ^{n×p} : Xᵀ·X = I_p}` (orthonormal p-frames in ℝⁿ).
- ProjectTangent: `P_X(V) = V − X·sym(Xᵀ·V)` where `sym(A) = (A + Aᵀ)/2`.
- Exp closed-form via matrix exponential (Edelman-Arias-Smith 1998 eq. 2.42): `Exp_X(V) = [X V]·exp([Xᵀ·V, −Vᵀ·V; I, Xᵀ·V])·[I; 0]`. Needs `linalg.MatrixExp` (sub-blocker, 097-T1, 092-T2.3-blocker).
- **QR retraction (cheap, 1st-order)**: `Retract_X(V) = Q` where `X + V = Q·R` (modified Gram-Schmidt). Already buildable on existing `linalg.QRDecompose`. ~30 LOC.
- **Polar retraction (cheap, 2nd-order)**: `Retract_X(V) = (X + V)·((X + V)ᵀ·(X + V))^{-½}` — the orthogonal-Procrustes solution closest to `X + V` in Frobenius norm. Reduces to `(X + V)·Σ^{-1}·U^T·U·V^T` via SVD; needs full SVD (currently absent — slot 097-T1 sub-blocker).
- **Cayley retraction (alternative)**: `Retract_X(V) = (I − ½·W(V))^{-1}·(I + ½·W(V))·X` with `W(V) = (I − ½·X·Xᵀ)·V·Xᵀ − X·Vᵀ·(I − ½·X·Xᵀ)`. Needs only matrix inverse. ~80 LOC.
- VectorTransport via QR-retraction differential: `T_{X → Y}(V) = Y·sym(Yᵀ·V) − Y·... ` (Absil-Malick 2012 *SIAM J Optim* 22:135); ~60 LOC.

Reference: Edelman-Arias-Smith 1998 *SIMAX* 20:303 (the canonical paper); AMS §3.6.2 + §8.1.3; Boumal 2023 §7.3.

(c) **LOC:** ~280 in `optim/manifold/stiefel.go`. The QR-retraction sub-path (~80 LOC) ships *unblocked* on existing `linalg.QRDecompose`; the Exp + polar-retraction path waits on SVD (097-T1) + MatrixExp (097-T2.3).

(d) **Partial blocker:** Exp closed-form needs `linalg.MatrixExp`; polar retraction needs SVD. QR retraction ships unblocked.

### R4 — Grassmann manifold Gr(n, p)

(a) **Ships:** nothing. `linalg.PCA` (in `linalg/pca.go`) is structurally a maximisation on Gr(n, p) (the leading p-dimensional principal subspace) but hardcodes the non-Riemannian power-iteration / SVD path.

(b) **Add:** `Gr(n, p) = St(n, p) / O(p)` — equivalence classes of orthonormal frames spanning the same p-plane; canonical representative as `n×p` matrix `X` with `Xᵀ·X = I_p` modulo right-action of O(p).
- Tangent space: `T_X Gr(n, p) = {Δ ∈ ℝ^{n×p} : Xᵀ·Δ = 0}` (horizontal lift of the Stiefel tangent to the quotient).
- Exp closed-form (Edelman-Arias-Smith 1998 eq. 2.65): `Exp_X(Δ) = [X·V cos(Σ) + U·sin(Σ)]·Vᵀ` where `Δ = U·Σ·Vᵀ` is the thin SVD of Δ.
- Log: `Log_X(Y) = U·atan(Σ)·Vᵀ` where `Y·(Xᵀ·Y)^{-1}·Xᵀ − X = U·Σ·Vᵀ` is the thin SVD of the deflation.
- Retract: SVD-based polar (subspace-aware), or QR with column-space normalisation.
- GeodesicDistance: `‖principal angles(X, Y)‖_2 = ‖atan(Σ)‖_2` from the deflation SVD.
- Principal angles via SVD: `Xᵀ·Y = U·diag(cos θ_i)·Vᵀ`; the `θ_i` are the principal angles between the subspaces spanned by columns of X and Y. ~30 LOC of glue once SVD ships.

Reference: Edelman-Arias-Smith 1998 §2.5; AMS §3.6.3 + §8.1.4; Boumal 2023 §9.

(c) **LOC:** ~250 in `optim/manifold/grassmann.go`. Sub-blocker on full SVD (eigenvector-returning, 097-T1) — without it, the Grassmann Exp/Log/Distance functions cannot be written in closed form. Workaround: the Stiefel substitute (R3) plus a quotient-by-O(p) post-processor reduces to Stiefel for many algorithms (gradient direction is unique modulo O(p) action; gradient norm and step-size unaffected).

(d) **Sub-blocker:** SVD with eigenvectors (097-T1).

### R5 — SPD manifold (symmetric positive-definite matrices)

(a) **Ships:** nothing. `linalg.CholeskyDecompose` exists — useful for the Bures-Wasserstein metric path (T3.8 in 092 spec) and for parameterising SPD as L·Lᵀ with L lower-triangular.

(b) **Add:** `SPD(n) = {P ∈ ℝ^{n×n} : P = Pᵀ, P ≻ 0}` with three distinct Riemannian metrics (each used by different consumers):

**Affine-invariant metric (AIM)** — Pennec-Fillard-Ayache 2006:
- `g_P(X, Y) = trace(P^{-1}·X·P^{-1}·Y)`.
- Exp: `Exp_P(X) = P^{1/2}·expm(P^{-1/2}·X·P^{-1/2})·P^{1/2}`.
- Log: `Log_P(Q) = P^{1/2}·logm(P^{-1/2}·Q·P^{-1/2})·P^{1/2}`.
- Distance: `d(P, Q) = ‖logm(P^{-1/2}·Q·P^{-1/2})‖_F`.
- Property: invariant under congruence `P ↦ Aᵀ·P·A` for invertible A — preserved by reparameterisations of the underlying Gaussian.

**Log-Euclidean metric** — Arsigny-Fillard-Pennec-Ayache 2006:
- Exp/Log map SPD ↔ `sym(n)` (the symmetric matrices vector space, flat); compute everything in log-coords; map back.
- `d_LE(P, Q) = ‖logm(P) − logm(Q)‖_F`. Cheap, commutative, but ignores the manifold's geodesic curvature. Suitable when SPD samples are clustered.

**Bures-Wasserstein metric** — Bhatia-Jain-Lim 2019, Malagò-Montrucchio-Pistone 2018:
- `d²_BW(P, Q) = trace(P + Q − 2·(P^{1/2}·Q·P^{1/2})^{1/2})` — the OT-on-Gaussians metric.
- The natural metric when SPD matrices are *covariance matrices* of Gaussians.
- Cross-link: when 092-T1.10 (JKO on simplex) lifts to JKO on Gaussians, this is the metric used.

Reference: Pennec-Sommer-Fletcher 2020 *Riemannian Geometric Statistics* §3 (the canonical edited volume); Bhatia-Jain-Lim 2019 *Expositiones Math* 37:165 (BW metric); Higham 2008 *Functions of Matrices* §10–§11 (the matrix-exp/log substrate).

(c) **LOC:** ~360 in `optim/manifold/spd.go`. Each metric is ~120 LOC (~30 LOC structural + ~90 LOC closed-form Exp/Log/Distance/PT).

(d) **Blocker:** AIM and BW need `linalg.MatrixExp`, `linalg.MatrixLog`, `linalg.MatrixSqrt` (097-T1 and 097-T2.3 sub-blockers). Log-Euclidean ships unblocked once MatrixExp/Log lands. SPD-AIM via Cholesky workaround for BW: parametrise `P = L·Lᵀ`; `d²_BW(L·Lᵀ, M·Mᵀ) = ‖L − M‖_F²`-equivalent only when `Lᵀ·M` is symmetric — falls short of the full metric.

### R6 — Hyperbolic manifold (Lorentz model + Poincaré ball)

(a) **Ships:** nothing.

(b) **Add:** `H^n = {x ∈ ℝ^{n+1} : ⟨x, x⟩_L = −1, x_0 > 0}` (Lorentz model) ↔ `B^n = {x ∈ ℝⁿ : ‖x‖ < 1}` (Poincaré ball) — two isometric models with closed-form bijection; ship both because consumers prefer different models for different reasons (Lorentz is numerically robust, no boundary singularity; Poincaré is intuitive and used by Nickel-Kiela 2017 for ML embeddings).
- **Lorentz Exp**: `Exp_x(v) = cosh(‖v‖_L)·x + sinh(‖v‖_L)/‖v‖_L·v` with the Minkowski inner product `⟨u,v⟩_L = −u_0·v_0 + Σ u_i·v_i`. ~40 LOC.
- **Lorentz Log**: `Log_x(y) = arccosh(−⟨x, y⟩_L) · (y + ⟨x,y⟩_L·x) / ‖·‖_L`. ~30 LOC.
- **Poincaré-ball gyrovector arithmetic**: Möbius addition `x ⊕ y = ((1 + 2·⟨x, y⟩ + ‖y‖²)·x + (1 − ‖x‖²)·y) / (1 + 2·⟨x, y⟩ + ‖x‖²·‖y‖²)`; Möbius scalar mul; Exp/Log via gyrovector formulas. ~80 LOC.
- **Bijection**: `Lorentz ↔ Poincaré` via stereographic projection. ~40 LOC.
- **Hyperbolic distance**: closed-form `arccosh(−⟨x, y⟩_L)` in Lorentz; equivalent gyrovector formula in Poincaré.

Reference: Ungar 2008 *Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity* (gyrovector approach); Nickel-Kiela 2017 NeurIPS (Poincaré embeddings); Pennec-Sommer-Fletcher 2020 §2.

(c) **LOC:** ~250 in `optim/manifold/hyperbolic.go`. Ships unblocked on stdlib.

(d) **No blocker.**

### R7 — Riemannian Gradient Descent (R-GD)

(a) **Ships:** Euclidean `GradientDescent` only. (b) **Add:** `x_{k+1} = Retract_{x_k}(−α_k·grad^M f(x_k))` with `grad^M f(x) = ProjectTangent_x(grad^Eucl f(x))`. Armijo line search adapted to manifold (compare `f(Retract_x(α·d))` vs `f(x) + c1·α·⟨grad^M, d⟩_x`). Reference: AMS §4; Boumal 2023 §4. (c) **LOC:** ~150 in `optim/manifold/rgd.go`. (d) **No blocker.**

### R8 — Riemannian Conjugate Gradient (R-CG)

(a) **Ships:** nothing. (b) **Add:** Sato-Iwai 2015 R-Polak-Ribière+ with vector transport: `β_k = max(0, ⟨grad_{k+1}, grad_{k+1} − VT(grad_k)⟩ / ⟨grad_k, grad_k⟩)`, `d_{k+1} = −grad_{k+1} + β_k·VT(d_k)`, `x_{k+2} = Retract_{x_{k+1}}(α_k·d_{k+1})`. Restart every `n` iters or when `⟨d, grad⟩ ≥ 0`. Reference: Sato-Iwai 2015 *Optimization* 64:1011. (c) **LOC:** ~180 in `optim/manifold/rcg.go`. (d) **Soft blocker:** Euclidean CG (102-T1.1, ~120 LOC) lands first.

### R9 — Riemannian L-BFGS (R-L-BFGS)

(a) **Ships:** Euclidean L-BFGS only. (b) **Add:** Huang-Gallivan-Absil 2015 R-L-BFGS — two-loop recursion lifts to manifolds via vector-transport of stored `s_k = Log_{x_k}(x_{k+1})` and `y_k = grad_{k+1} − VT(grad_k)` pairs. Cautious update: skip pair if `⟨s_k, y_k⟩_{x_k} ≤ 0`. Reference: Huang-Gallivan-Absil 2015 *SIAM J Optim* 25:1660. (c) **LOC:** ~280 in `optim/manifold/rlbfgs.go` (larger than Euclidean — VT-of-history bookkeeping). (d) **No blocker** beyond R1.

### R10 — Riemannian Trust Region (R-TR)

(a) **Ships:** nothing. (b) **Add:** AMS Algorithm 10 — local quadratic model `m_k(η) = f(x_k) + ⟨grad^M, η⟩_{x_k} + ½·⟨H_k·η, η⟩_{x_k}` on `T_{x_k} M`; solve `argmin_{η ∈ T_{x_k} M, ‖η‖ ≤ Δ_k} m_k(η)` via truncated-CG (Steihaug-Toint) on tangent space; ratio `ρ = (f(x_k) − f(Retract(η)))/(m_k(0) − m_k(η))` for accept/reject + radius update. *Globally convergent* without line search; superlinear when Hessian exact. Reference: AMS §7 (Manopt-MATLAB reference); Boumal-Absil-Cartis 2019 *IMA J Numer Anal*. (c) **LOC:** ~320 in `optim/manifold/rtr.go`. (d) **Soft blocker:** Euclidean Steihaug-CG (102-T1.3, ~200 LOC).

### R11 — Riemannian Hessian + Levi-Civita connection

(a) **Ships:** nothing. `autodiff/` ships reverse-mode scalar tape only — no HVP. (b) **Add:** Hess-on-tangent construction: `Hess^M f(x)·v = P_x(D(grad^M f)(x)·v) − Weingarten_x(v, P⊥_x(grad^Eucl f(x)))`. Second term is the shape correction (nonzero only when normal component of Euclidean gradient is nonzero); for sphere, `Weingarten_x(v, w_n) = ⟨w_n, x⟩·v`. Christoffel symbols *implicit* in embedded-submanifold construction (never materialised — the whole point vs intrinsic charts). Closed-form Hess per manifold: Sphere ~20 LOC, Stiefel ~40 (Edelman 1998 eq. 2.53), Grassmann ~40 (eq. 2.71), SPD-AIM ~50 (Pennec 2006), Hyperbolic ~30. Reference: AMS §5; Boumal 2023 §5.6 + §10.2. (c) **LOC:** ~180 in `optim/manifold/hess.go`. Generic autodiff fallback ~80 LOC blocked on HVP (012-T1). (d) **Soft blocker** for generic; closed-form per-manifold ships unblocked.

### R12 — Riemannian Newton + damped Newton

(a) **Ships:** nothing (1-D Euclidean only). (b) **Add:** R-Newton solves `Hess^M f(x_k)·η_k = −grad^M f(x_k)` on `T_{x_k} M`, then `x_{k+1} = Retract_{x_k}(η_k)`. Tangent-space CG iteration (R-TR inner loop without trust-region constraint). Damped variant: `(Hess + λ·I_{T_x M})·η = −grad` — useful at saddle points (common on curved manifolds). Reference: AMS §6; Boumal 2023 §10. (c) **LOC:** ~150 in `optim/manifold/rnewton.go`. (d) **No blocker** beyond R11.

### R13 — Riemannian SGD (R-SGD) and stochastic Riemannian methods

(a) **Ships:** nothing. `optim/` doesn't ship plain Euclidean SGD either (102-T1.8).

(b) **Add:** the modern stochastic-on-manifold canon:
- **R-SGD** (Bonnabel 2013 *IEEE TAC* 58:2217): `x_{k+1} = Retract_{x_k}(−η_k · grad_minibatch^M f(x_k))`. Almost-sure convergence to critical points under standard step-size conditions.
- **R-Adam / R-AdaGrad** (Kasai-Sato 2019, Bécigneul-Ganea ICLR 2019): adaptive learning rate on each tangent direction; preconditioner stored *on tangent space* and vector-transported between iterates.
- **R-SVRG** (Zhang-Reddi-Sra NeurIPS 2016): variance-reduced R-SGD with periodic full-batch gradient.
- **R-momentum** (Alimisis-Bécigneul-Ganea-Lucchi-Hofmann ICLR 2020): heavy-ball with vector transport.

Reference: Bonnabel 2013 (the foundational R-SGD paper); Kasai-Sato 2019 *NeurIPS* (R-Adam); Zhang-Reddi-Sra 2016 *NeurIPS* (R-SVRG).

(c) **LOC:** ~280 in `optim/manifold/rsgd.go` (R-SGD + R-Adam + R-momentum). R-SVRG +120 LOC if needed.

(d) **No blocker** beyond R1. Soft cross-link to Euclidean Adam/SGD shipping in 102-T1.6/T1.8.

### R14 — Riemannian-manifold HMC (RMHMC) [pairs with 092-T3.12]

(a) **Ships:** nothing. `prob/` ships standard HMC? Verify — actually `prob/` ships distributions only (per 117-prob-missing); HMC is missing.

(b) **Add:** Girolami-Calderhead 2011 *JRSS B* 73:123 — uses the local Fisher metric as the HMC mass matrix, allowing efficient sampling on highly-correlated posteriors. Pairs with 092-T1.2 (Fisher information for Gaussian) + 092-T3.12 + 204-symplectic (the leapfrog inside HMC must be the *generalised* leapfrog for non-constant mass matrix, equivalent to a symplectic step on a non-Euclidean manifold).

(c) **LOC:** ~250 in `optim/manifold/rmhmc.go`. Pairs with `infogeo/fisher.go` (092-T1.2) and `chaos/symplectic.go` (204-keystone).

(d) **Hard blocker:** chaos/symplectic.go (204-keystone, ~140 LOC) + infogeo/fisher.go (092-T1.2, ~80 LOC).

### R15 — Karcher / Fréchet mean on a manifold

(a) **Ships:** nothing. (Quaternion average / pose averaging deferred per 205-L13, 150 LOC, scoped for SO(3) only.)

(b) **Add:** the generic iterative algorithm:
```
μ_0 = arbitrary x_0  (or chordal mean for closed-form initialisation when available)
loop until convergence:
    δ = (1/N) · Σ_i Log_{μ}(x_i)
    μ ← Retract_μ(δ)         // or Exp_μ(δ) for full second-order convergence
```
Quadratic convergence when samples are within injective radius (per 205-L13 R6). Generic for any `RiemannianMetric` — ships once and works on Sphere/Stiefel/Grassmann/SPD/Hyperbolic/SO(3)/SE(3).

Karcher variance: `(1/N) Σ d²(μ, x_i)` — Riemannian variance, used as a goodness-of-fit score for distributions on manifolds.

Reference: Karcher 1977 *CPAM* 30:509; Pennec 1998; Moakher 2002 *SIMAX* 24:1; Manton 2004 *J ACSSC*.

(c) **LOC:** ~100 in `optim/manifold/karcher.go`. Generic — *one* implementation works across all manifolds via the interface. 205-L13's SO(3)/SE(3) Karcher mean reduces to instantiating this generic on the SO(3)/SE(3) `Manifold`.

(d) **No blocker.** Pairs naturally with 205-L13.

### R16 — Riemannian PCA / tangent-space PCA

(a) **Ships:** `linalg.PCA` (linear PCA in ℝⁿ). (b) **Add:** Fletcher-Lu-Pizer-Joshi 2004 — Karcher mean μ, project samples to `T_μ M` via Log, run linear PCA on `{v_i = Log_μ(x_i)}`, lift via Exp. Used: shape statistics, diffusion-tensor imaging (SPD tangent PCA), pose stats (tangent PCA on SE(3)). Pennec 2018 *Barycentric Subspace Analysis* generalises further. Reference: Fletcher et al. 2004 *IEEE TMI* 23:995. (c) **LOC:** ~120 in `optim/manifold/rpca.go` (consumes R15 + linalg.PCA). (d) **No blocker.**

### R17 — Procrustes / ICP

(a) **Ships:** nothing (177-SG5 flagged ~120-LOC connective). (b) **Add:** orthogonal Procrustes `argmin_{R ∈ O(n)} ‖A·R − B‖_F` closed form via SVD: `Aᵀ·B = U·Σ·Vᵀ`, `R = V·Uᵀ`. Constrained-rotation case (det = +1) for rigid registration. ICP (Besl-McKay 1992): alternating correspondence + Procrustes — manifold-optim in disguise. Reference: Schönemann 1966; Besl-McKay 1992 *IEEE PAMI* 14:239. (c) **LOC:** ~80 in `optim/manifold/procrustes.go`. (d) **Sub-blocker:** SVD (097-T1).

### R18 — Online subspace tracking on Stiefel

(a) **Ships:** nothing. (b) **Add:** **GROUSE** (Balzano-Recht-Nowak 2010) — streaming PCA on Gr(n, p) via rank-1 retractions on Stiefel. Used: video background subtraction, online matrix completion, partial-observation subspace tracking. **PETRELS** (Chi-Eldar-Calderbank 2013) — RLS-based rank-`p` extension. Reference: Balzano 2010 *Allerton*; Chi 2013 *IEEE TSP* 61:5947. (c) **LOC:** ~150 in `optim/manifold/grouse.go`. (d) **No blocker.**

### R19 — Geodesic shooting + geodesic ODE

(a) **Ships:** nothing. RK4 in `chaos/` only — non-symplectic, drifts off manifold. (b) **Add:** for manifolds without closed-form Log: (i) IVP `Exp` via Verlet (204-keystone) on geodesic Hamiltonian `H = ½·g^{ij}(q)·p_i·p_j`, ~80 LOC; (ii) BVP `Log` via Newton on residual `Exp_p(v) − q = 0`, ~80 LOC; (iii) mesh geodesics via heat method (Crane-Weischedel-Wardetzky 2013), pairs with 177-SG13. Reference: AMS §3. (c) **LOC:** ~250 in `optim/manifold/geodesic_ode.go`. (d) **Hard blocker:** chaos/symplectic.go (204-keystone).

### R20 — Vector transport

(a) **Ships:** nothing. (b) **Add:** three schemes — differentiated retraction (`T_{x→y}(v) = D(Retract_x)(v) → tangent at y`, ~30 LOC closed-form per manifold); Schild's ladder generic 4-step (Lorenzi-Pennec 2014, ~80 LOC); pole ladder (cheaper variant). Closed-form PT: Sphere ~20, Stiefel ~50 (Edelman 1998), Grassmann ~50, SPD-AIM ~30 (`PT_{P→Q}(X) = (Q·P^{-1})^{1/2}·X·(P^{-1}·Q)^{1/2}`), Hyperbolic ~40. Reference: AMS §8; Lorenzi-Pennec 2014 *IJCV* 105:111. (c) **LOC:** ~180 in `optim/manifold/transport.go`. (d) **No blocker** for closed-form path.

### R21 — Level-set / constraint manifold

(a) **Ships:** nothing as manifold (`proximal.ProxLinear` projects onto affine subspace, Euclidean). (b) **Add:** for smooth `c: ℝⁿ → ℝᵏ` with full-rank `Jc(x)`, the manifold `M = {x : c(x) = 0}`: `ProjectTangent_x(v) = v − Jcᵀ·(Jc·Jcᵀ)^{-1}·Jc·v`; retract via 1-2 Newton steps on c. Closed-form for affine `Ax=b` (one-step exact); closed-form for quadric `xᵀAx=c`. Reference: AMS §3.6. (c) **LOC:** ~200 in `optim/manifold/levelset.go`. (d) **No blocker.**

### R22 — Riemannian Langevin / SGLD

(a) **Ships:** nothing. SDE canon absent (slot 202-new-sde). (b) **Add:** `dX_t = −grad^M U(X_t)·dt + √(2T)·dB^M_t` where `dB^M_t` is Brownian motion on M (Stratonovich projection from ambient). Discretisation: Riemannian Euler-Maruyama with retraction. Used: Bayesian sampling on simplex via Fisher-Rao, molecular dynamics on Lie groups, SO(3)/Stiefel orientation posteriors. Reference: Brubaker-Salzmann-Urtasun 2012; Liu-Zhu-Ramadge 2016 *NeurIPS*. (c) **LOC:** ~180 in `optim/manifold/langevin.go`. (d) **Hard blocker:** SDE canon (202).

---

## 2. Implementation-detail summary table

| ID | Primitive | LOC | File | Reference |
|----|-----------|-----|------|-----------|
| R1 | `Manifold` / `RiemannianMetric` interface | 120 | optim/manifold/manifold.go ★ | AMS §3 |
| R2 | Sphere S^{n-1} | 140 | optim/manifold/sphere.go ★ | Edelman 1998 §2.2 |
| R3 | Stiefel St(n,p) + QR/polar/Cayley retractions | 280 | optim/manifold/stiefel.go | Edelman 1998 §2.4 |
| R4 | Grassmann Gr(n,p) | 250 | optim/manifold/grassmann.go | Edelman 1998 §2.5 |
| R5 | SPD with AIM + Log-Euclidean + Bures-Wasserstein | 360 | optim/manifold/spd.go | Pennec 2006; Bhatia 2019 |
| R6 | Hyperbolic Lorentz + Poincaré ball | 250 | optim/manifold/hyperbolic.go | Ungar 2008; Nickel-Kiela 2017 |
| R7 | Riemannian gradient descent | 150 | optim/manifold/rgd.go ★ | AMS §4 |
| R8 | Riemannian conjugate gradient | 180 | optim/manifold/rcg.go | Sato-Iwai 2015 |
| R9 | Riemannian L-BFGS | 280 | optim/manifold/rlbfgs.go | Huang et al. 2015 |
| R10 | Riemannian trust region | 320 | optim/manifold/rtr.go | AMS §7 |
| R11 | Riemannian Hessian + Weingarten | 180 | optim/manifold/hess.go | AMS §5 |
| R12 | Riemannian Newton + damped Newton | 150 | optim/manifold/rnewton.go | AMS §6 |
| R13 | R-SGD / R-Adam / R-momentum / R-SVRG | 280 | optim/manifold/rsgd.go | Bonnabel 2013 |
| R14 | Riemannian-manifold HMC | 250 | optim/manifold/rmhmc.go | Girolami-Calderhead 2011 |
| R15 | Karcher / Fréchet mean | 100 | optim/manifold/karcher.go | Karcher 1977; Pennec 1998 |
| R16 | Riemannian PCA / tangent PCA | 120 | optim/manifold/rpca.go | Fletcher 2004 |
| R17 | Procrustes on Stiefel / orth. Procrustes | 80 | optim/manifold/procrustes.go | Schönemann 1966 |
| R18 | GROUSE / PETRELS streaming subspace | 150 | optim/manifold/grouse.go | Balzano 2010 |
| R19 | Geodesic shooting + geodesic ODE | 250 | optim/manifold/geodesic_ode.go | AMS §3 |
| R20 | Vector transport (Schild's + closed-forms) | 180 | optim/manifold/transport.go | Lorenzi-Pennec 2014 |
| R21 | Level-set constraint manifold | 200 | optim/manifold/levelset.go | AMS §3.6 |
| R22 | Riemannian Langevin / SGLD | 180 | optim/manifold/langevin.go | Liu-Zhu-Ramadge 2016 |
|    | **Total core (R1 + R2 + R7 + R15 + R20)** | **~690** | | |
|    | **Total Tier-1 ship-now (R1+R2+R3-QR+R7+R15+R20-closedform)** | **~960** | | |
|    | **Total full canon** | **~3,200** | | |

★ = keystone (R1 interface validates entire `Manifold` contract; R2 sphere is the simplest closed-form instance; R7 R-GD is the simplest optimiser).

---

## 3. Tier ordering (ship sequence)

**Tier 1 (480 LOC, ship in 1 sprint, validates the entire interface):**
1. R1 `Manifold` / `RiemannianMetric` interface (120 LOC). Foundation.
2. R2 Sphere S^{n-1} with closed-form everything (140 LOC). The "hello world" manifold.
3. R7 Riemannian Gradient Descent + Armijo line search (150 LOC). Simplest optimiser.
4. R20-partial closed-form sphere PT (30 LOC). Required by R8/R9/R13/R15 once they ship.
5. R3-partial Stiefel with QR retraction only (80 LOC, ships on existing `linalg.QRDecompose`). Validates the non-trivial-but-still-closed-form case.

After Tier 1: the `Manifold` interface is callable, sphere optimisation works end-to-end, the Stiefel partial-surface (without Exp closed-form, without polar retraction) handles ~80% of practical Stiefel use cases (online PCA, neural-network orthogonal-weight constraints), and the cross-language parity contract is shippable on the sphere.

**Tier 2 (810 LOC, ship 2nd sprint, completes the textbook):**
6. R6 Hyperbolic Lorentz + Poincaré (250 LOC). Ships unblocked.
7. R8 R-CG (180 LOC). Soft cross-link to Euclidean CG (102-T1.1).
8. R11 Riemannian Hessian + closed-form per-manifold (180 LOC).
9. R12 R-Newton + damped Newton (150 LOC).
10. R15 Karcher / Fréchet mean (100 LOC). Generic — works on every Tier 1 manifold once interface exists.

After Tier 2: full first- and second-order optimisation works on Sphere/Hyperbolic/Stiefel-partial. Karcher mean unifies 205-L13 (SO(3)/SE(3) special case).

**Tier 3 (820 LOC, ship 3rd sprint, advanced manifolds):**
11. R9 R-L-BFGS (280 LOC).
12. R10 R-TR (320 LOC). Soft blocker on Euclidean Steihaug-CG (102-T1.3).
13. R5-partial SPD with Log-Euclidean only (120 LOC) — ships once `linalg.MatrixExp` + `linalg.MatrixLog` land.
14. R16 Riemannian PCA / tangent PCA (120 LOC).

After Tier 3: SPD-via-LogEuclidean works; tangent-PCA on every shipped manifold; second-order optimisation matches scipy / Pymanopt feature parity.

**Tier 4 (~570 LOC, ship-when-consumer-pulls):**
15. R4 Grassmann (250 LOC). Sub-blocker on full SVD (097-T1).
16. R3-full Stiefel with closed-form Exp + polar retraction (200 LOC). Sub-blocker on SVD + MatrixExp.
17. R5-full SPD with AIM + Bures-Wasserstein (240 LOC). Sub-blocker on MatrixSqrt.
18. R17 Procrustes / ICP (80 LOC). Sub-blocker on SVD.

**Tier 5 (~860 LOC, deferred):**
19. R13 R-SGD canon (280 LOC). Pairs with Euclidean SGD/Adam (102-T1.6/T1.8).
20. R14 RMHMC (250 LOC). Hard blocker on chaos/symplectic + infogeo/fisher.
21. R18 GROUSE (150 LOC). Niche.
22. R19 geodesic shooting (250 LOC). Hard blocker on chaos/symplectic.
23. R21 level-set constraint (200 LOC). Niche.
24. R22 Riemannian Langevin (180 LOC). Hard blocker on full SDE canon (202).

---

## 4. Architectural recommendations

**A1. New sub-package `optim/manifold/`.** Mirrors the precedent established by `optim/proximal/` and `optim/transport/` — a sub-package of `optim/` for a substantial sub-canon with its own internal structure. ~3,200 LOC at full canon, ~960 LOC at Tier 1. One file per manifold (sphere.go, stiefel.go, grassmann.go, spd.go, hyperbolic.go) + one file per algorithm (rgd.go, rcg.go, rlbfgs.go, rtr.go, karcher.go, ...) + interface (manifold.go) + transport (transport.go).

**A2. Cross-package recommended consumption pattern.** `infogeo/manifold_fisher.go` (092-T2.1–T2.7) imports `optim/manifold/sphere.go` (R2) to get the Fisher-Rao simplex via the `√p` sphere-pullback embedding. `geometry/so3.go` (205-keystone) ships SO(3) as a concrete `Manifold` (implements `optim/manifold.Manifold`). `geometry/se3.go` (205-L6) does the same for SE(3). `optim/manifold/karcher.go` (R15) becomes the *single* implementation of Karcher mean, used by 205-L13 (SO(3)/SE(3) means). No duplicate code across packages.

**A3. Cycle-free dependency DAG.**
```
optim/manifold/  →  {linalg, optim (parent for line-search etc.)}
infogeo/  →  optim/manifold/
geometry/  →  optim/manifold/  (when 205 ships SO(3)/SE(3) as Manifolds)
optim/  ↛  optim/manifold/   (parent never imports child sub-package)
```

**A4. Closed-form per-manifold path always preferred over generic numerical.** The repo's "reimplement from first principles" rule (CLAUDE.md §6) here means: for sphere, prefer hand-written `cos·x + sin/‖v‖·v` over the generic `Verlet-on-geodesic-ODE` path. Generic numerical fallback is last-resort, used only when no closed form exists (e.g., generic constraint manifold from R21).

**A5. Cross-language parity contract: round-trip + retraction-equivalence + manifold-axiom tests.** Required golden files per manifold:
- `sphere_roundtrip.json`: 200 vectors covering tangent-space norms ‖v‖ ∈ {0, 1e-6, 1e-3, 0.1, π/4, π/2, π−1e-6}; check `‖Log_x(Exp_x(v)) − v‖ ≤ 1e-12`.
- `sphere_metric_axioms.json`: 100 random pairs (x, y); check d(x, y) = d(y, x), d(x, x) = 0, triangle inequality d(x, z) ≤ d(x, y) + d(y, z) within injectivity radius.
- `stiefel_qr_retraction.json`: 100 random `(X, V)` pairs with `Xᵀ·X = I`; check `Retract_X(V)ᵀ · Retract_X(V) = I_p` to 1e-13.
- `karcher_mean_consistency.json`: 50 sample sets on each manifold; check the first-order condition `‖Σ_i Log_μ(x_i)‖ < 1e-10` at the converged mean.
- `geodesic_constancy.json`: 30 endpoint pairs per manifold; check `‖dExp_x(t·v)/dt‖_x` is constant in `t` (geodesic constant-speed property) to 1e-9.

**A6. Numerical-stability mandate on Taylor fallbacks.** Every `‖v‖ → 0` fallback (sphere Exp, Lorentz Exp, hyperbolic Log, etc.) MUST document its switching threshold (e.g., "‖v‖ < 2^{1/3}·sqrt(eps_machine) ≈ 6e-6 → use sin(‖v‖)/‖v‖ Taylor to order ‖v‖⁴"). Cross-language parity is brittle if Go uses 1e-4 and C++ uses 1e-8.

**A7. Zero-alloc hot path mandate.** All retraction / projection / Exp / Log functions consume caller-provided `out []float64` buffers. The R-GD inner loop iterates these thousands of times per Pistachio frame.

---

## 5. Risks / gotchas

**R1. Retraction-vs-Exp choice has order-of-magnitude wall-clock impact.** Per slot 095-§3 (the perf audit on infogeo's Riemannian roadmap): the retraction is 3-8× cheaper than Exp on SPD/Stiefel. Default to Retract everywhere; expose Exp as a parallel API for users who *need* the geodesic (e.g., Karcher mean prefers Exp for quadratic convergence; vanilla R-GD prefers Retract for 8× speedup at unchanged O(1/k) convergence rate).

**R2. Vector transport vs parallel transport: pick one default.** Closed-form parallel transport is preferred on Sphere/Stiefel/Grassmann/SPD-AIM/Hyperbolic (all have closed-form). Generic Schild's-ladder is the fallback. R-CG and R-L-BFGS work with either; R11 Hessian needs the parallel transport for the connection-coefficient correctness. Document the choice per algorithm.

**R3. Non-uniqueness on Grassmann: choice of representative.** A point on Gr(n, p) is an *equivalence class* of n×p matrices modulo right-action of O(p). All operations must be O(p)-invariant; otherwise the result depends on the arbitrary representative. Test contract: pick two representatives X, X·R for R ∈ O(p); verify `f(X) = f(X·R)` for any user-facing function f.

**R4. Antipodal singularity on Sphere at θ = π.** `Log_x(−x)` is direction-ambiguous (any unit vector orthogonal to x is a valid pre-image). Document the convention (return an arbitrary tangent direction, possibly NaN). Cross-language parity tests at θ ≈ π must allow ±v family of directions.

**R5. Stiefel polar vs QR vs Cayley retractions are not interchangeable.** Polar (SVD-based) is 2nd-order accurate (matches Exp to O(‖V‖²) at small V) and is the default in Manopt; QR is 1st-order; Cayley is 2nd-order but requires invertibility of `I − ½·W(V)`. Performance: QR ~2n·p² flops, polar ~6n·p² + SVD overhead, Cayley ~6n·p² + matrix-inverse overhead. Default: QR on a per-call basis when the result is consumed by R-GD; polar for R-Newton inner solver where 2nd-order accuracy improves convergence.

**R6. SPD manifold metric choice is consumer-dependent.** AIM: invariant under congruence; preferred for diffusion-tensor imaging. Log-Euclidean: cheap (no MatrixSqrt in inner loop); preferred when SPD samples are clustered. Bures-Wasserstein: OT-on-Gaussians; preferred when SPD matrices are *covariance matrices* and the underlying Gaussian-distribution geometry matters. Ship all three; document.

**R7. R-L-BFGS history transport overhead.** Naïve VT-of-history at each iteration is O(m·n) extra cost (m = history depth). Deferred-transport (transport only when accessed) reduces amortised cost; documented in Huang-Gallivan-Absil 2015 §5. Implementation choice: deferred-transport, with cached transport pointers for the (s_k, y_k) pairs.

**R8. Constraint-manifold (R21) requires Jc full-rank.** When `Jc(x)` rank-deficient (degenerate constraint normal), `(Jc·Jcᵀ)^{-1}` is singular and the projection fails. Document the precondition; consider damped pseudo-inverse `Jcᵀ·(Jc·Jcᵀ + λ·I)^{-1}` as a fallback (returns nearest-feasible-direction).

**R9. R-TR sub-problem is non-trivial.** Steihaug-CG truncated at trust-region boundary; the `‖η‖_x ≤ Δ` constraint uses the Riemannian metric, not the ambient Euclidean metric. Implementations that use the Euclidean norm get wrong step sizes. Test contract: at the same iterate, the Steihaug step on a non-flat manifold (e.g., sphere) must differ from the corresponding Euclidean step.

**R10. Cross-package coupling via `Manifold` interface adds surface area.** Once `optim/manifold.Manifold` is an exported interface, `geometry/so3.go`, `geometry/se3.go`, and `infogeo/manifold_fisher.go` will all implement it. Interface stability becomes a versioning concern. Mitigation: add interface methods conservatively; use explicit `BaseManifold` mixin for default implementations to allow non-breaking interface growth.

---

## 6. Cross-package coupling

| Edge | LOC | Purpose |
|------|-----|---------|
| optim/manifold/ → linalg/ (QR for Stiefel-QR-retract) | 0 (call-only) | R3 partial |
| optim/manifold/ → linalg/ (SVD for polar retract / Grassmann / Procrustes) | 0 (call-only) | R3-full / R4 / R17, **blocked on 097-T1** |
| optim/manifold/ → linalg/ (MatrixExp/Log/Sqrt for SPD AIM/BW) | 0 (call-only) | R5, **blocked on 097-T2.3** |
| optim/manifold/ → optim/ (line-search infrastructure) | 30 | R7/R9 reuse Armijo from `optim/gradient.go::lbfgsLineSearch` |
| infogeo/manifold_fisher.go → optim/manifold/sphere.go | 0 (consumer) | 092-T2.4 simplex via √p sphere |
| geometry/so3.go (205-keystone) → optim/manifold/manifold.go | 0 | 205-L13 + 205-L15 implement the Manifold interface |
| geometry/se3.go (205-L6) → optim/manifold/manifold.go | 0 | 205-L13 + 205-L15 |
| optim/manifold/karcher.go → 205-L13 SO(3)/SE(3) Karcher | 0 | 205-L13 reduces to instantiation |
| optim/manifold/rmhmc.go → infogeo/fisher.go (092-T1.2) | 0 | R14 mass matrix |
| optim/manifold/rmhmc.go → chaos/symplectic.go (204-keystone) | 0 | R14 leapfrog |
| optim/manifold/geodesic_ode.go → chaos/symplectic.go | 0 | R19 |
| optim/manifold/langevin.go → ?-sde (202) | 0 | R22 |
| optim/manifold/ → testdata/manifold_*.json | n/a | Cross-language parity grids per manifold |

Total connective LOC across edges: ~30 (only the line-search reuse from `optim/`). All other edges are *call-only* (no shared internal types). The Riemannian-optimisation module is unusually self-contained — most inputs/outputs are `[]float64` slices with the `Manifold` instance carrying the metric structure.

---

## 7. Single-highest-leverage 1-day project

**Tier-1 items R1 + R2 + R7 (+ R20-sphere-PT) = `optim/manifold/{manifold.go, sphere.go, rgd.go, transport.go}` core (~440 LOC).** Justification:

1. **Closes the entire "Riemannian optimisation" gap with one PR.** Today, `argmin f(x) s.t. x ∈ S^{n-1}` requires the user to roll their own projection; tomorrow it's a 5-line call to `RiemannianGradientDescent(NewSphere(n), f, gradEucl, x0, cfg)`.

2. **Validates the entire Manifold interface on the simplest closed-form case.** Sphere has *closed-form everything* (Exp, Log, PT, Distance, Retract, ProjectTangent). If the interface shape works for sphere, it works for every other manifold modulo manifold-specific arithmetic.

3. **Pure additive surface.** No break to existing `optim/`, `geometry/`, or `infogeo/`. All new identifiers under `optim/manifold/`.

4. **Unblocks 092-T2.4** (Fisher-Rao simplex via `√p` embedding into the sphere) — this slot's R2 IS 092-T2.4's underlying primitive.

5. **Unblocks 205-L13** (Karcher mean on SO(3)/SE(3)): once R15 ships (Tier 2), 205-L13 becomes a 30-LOC instantiation of R15 against 205-L1–L4's SO(3) Manifold implementation.

6. **Cross-language parity test contract is obvious.** Sphere-roundtrip Exp/Log + sphere-metric-axioms (symmetry, triangle inequality, geodesic constant-speed) — three golden-file grids, ~280 vectors, 1e-12 tolerance. Reproducible from first-principles in any language.

---

## 8. Single-highest-leverage cutting-edge piece

**Tier-2 R5-full (SPD with AIM + Bures-Wasserstein, 240 LOC) paired with R16 Riemannian PCA (120 LOC) = ~360 LOC for "geometry on covariance matrices."** Justification:

1. **Genuine cutting-edge with broad applicability.** SPD-Riemannian geometry (Pennec 2006) is the foundational primitive for: diffusion-tensor imaging (every brain-imaging pipeline since 2010), robust covariance estimation under the AIM metric, riemannian-PCA on multivariate-Gaussian families, Bures-Wasserstein-OT on Gaussians (the OT-on-distributions metric when the distributions are Gaussian), federated covariance averaging.

2. **No mainstream library ships all three SPD metrics cleanly with a unified API.** Pymanopt has SPD with AIM only. Geomstats has all three but is Python-only. R-package ships nothing. Reality would be the only zero-dependency pure-math library shipping AIM + Log-Euclidean + Bures-Wasserstein on SPD with the cross-language golden-file contract.

3. **Pairs with 092-T2.3** (SPD as a statistical manifold for Gaussians) — sharing the same closed-form implementation. 092 punted SPD to v2 because of the MatrixExp/Log sub-blocker; this slot's R5 specifies what the 092 v2 looks like.

4. **Unblocks Bures-Wasserstein gradient flow on Gaussians.** Combined with `optim/transport/` (already ships Sinkhorn) and R5-BW, the JKO scheme on Gaussians becomes implementable — the canonical "Wasserstein gradient flow on a parametric family" demonstration.

5. **Test contract is strong.** SPD round-trip Exp/Log under each of three metrics; congruence-invariance test for AIM (`d_AIM(A·P·Aᵀ, A·Q·Aᵀ) = d_AIM(P, Q)` for any invertible A); BW-equals-Frobenius-on-commuting-SPD test (`d_BW(P, Q) = ‖P^{1/2} − Q^{1/2}‖_F` when P, Q commute).

---

## 9. Verdict

**SHIP** Tier 1 (~480 LOC over 1 sprint) — `Manifold` interface + Sphere + R-GD + sphere PT. Validates the entire architecture on the simplest closed-form case; unblocks 092-T2.4 and 205-L13.

**SHIP** Tier 2 (~810 LOC over 2nd sprint) — Hyperbolic + R-CG + Riemannian Hessian + R-Newton + Karcher mean. Soft cross-link to 102-T1.1 (Euclidean CG). Brings the R-optim canon to feature parity with Pymanopt's first-and-second-order methods on Sphere/Hyperbolic/Stiefel-partial.

**SHIP** Tier 3 (~820 LOC over 3rd sprint) — R-L-BFGS + R-TR + SPD-Log-Euclidean + tangent PCA. Soft blockers on 102-T1.3 (Steihaug-CG) and 097-T2.3 (MatrixExp/Log). Brings the canon to Manopt-MATLAB feature parity.

**SHIP-WHEN-CONSUMER-PULLS** Tier 4 (~570 LOC) — Grassmann + Stiefel-full + SPD-AIM/BW + Procrustes. Sub-blocked on full SVD (097-T1).

**DEFER** Tier 5 (~860 LOC) — R-SGD canon + RMHMC + GROUSE + geodesic shooting + level-set + Langevin. Hard blockers on chaos/symplectic (204), full SDE canon (202), and the Euclidean-Adam/SGD substrate (102-T1.6/T1.8). Defer until consumer pulls.

**Cross-slot synergy callouts:**
- **092-infogeo-missing T2.1–T2.7**: the *statistical-manifold* parallel scope. Recommend: 092-T2.1 imports this slot's R1; 092-T2.4 (sphere) IS this slot's R2; 092-T2.5 (hyperbolic) IS this slot's R6; 092-T2.6 (Stiefel/Grassmann) IS this slot's R3+R4. Single source of truth for each manifold; metric-decoration in `infogeo/`.
- **205-new-lie-groups L13 + L15**: 205-L13 (Karcher mean on SO(3)/SE(3)) reduces to instantiation of R15. 205-L15 (Lie-Newton / Lie-GD / ESM on SO(3)/SE(3)) reduces to instantiation of R7+R12 with SO(3)/SE(3) `Manifold` implementations.
- **177-synergy-geometry-optim SG5/SG6 (Procrustes/ICP)**: directly reduces to R17 once SVD lands.
- **102-optim-missing T1.1/T1.3**: Euclidean CG and Euclidean Steihaug-CG are *substrate* for R8/R10. Soft cross-links — landing them first cleans up the implementation but is not a hard blocker.
- **097-linalg-missing T1**: full SVD with eigenvectors is the **single most-cited cross-package blocker** from this slot (gates Stiefel-full, Grassmann, SPD-AIM, SPD-BW, Procrustes — five of twenty-two primitives). The MatrixExp/Log/Sqrt cluster (097-T2.3) gates SPD-AIM/BW further.
- **204-new-symplectic-int**: chaos/symplectic.go (204-keystone) is required for R14 (RMHMC) and R19 (geodesic ODE). Hard blocker.
- **202-new-sde**: the entire SDE canon (~1,500 LOC) is required for R22 (Riemannian Langevin). Hard blocker; deferral expected.
- **011/012/013-autodiff**: forward-mode + HVP is required for the *generic-numerical* path of R11 (Riemannian Hessian via autodiff over the gradient closure); the closed-form per-manifold path ships unblocked.

---

*206-new-riemannian-opt.md — 342 lines.*

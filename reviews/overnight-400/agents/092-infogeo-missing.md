## 092 — `infogeo/` missing primitives

**Topic.** α-connections, dual flat coordinates, e/m projections, mixture-family geodesics, natural gradient, Wasserstein gradient flows.
**Package.** `C:\limitless\foundation\reality\infogeo\` (4 src files: `doc.go`, `fdiv.go`, `bregman.go`, `mmd.go`; 1,373 LOC total).
**Date.** 2026-05-07.
**Frame.** 091 already pinned the headline finding — the package is named `infogeo` but ships *information-theoretic divergences on the simplex*, not *information geometry on a Riemannian manifold*. This audit enumerates what *would* have to land for the name to be honest, ranked by leverage and dependency-order.

External reference frame:
- **Geomstats** (Miolane et al., JMLR 2020; v2.7 2025): Python lib of choice for Riemannian/IG. ~30 manifolds, full `exp`/`log`/`parallel_transport`/`christoffels`/`metric_matrix` API. The contract this package would extend if it grew.
- **Pymanopt** (Townsend et al., JMLR 2016; v2.x 2024): manifold optimisation. SPD / Stiefel / Grassmann / sphere / Poincaré with retractions + vector transports.
- **Amari S. (2016).** *Information Geometry and its Applications*. Applied Mathematical Sciences vol 194. Springer. The canonical citation for α-connections, dual flat structure, e/m projections, mixture/exponential family geodesics.
- **Nielsen, F. (2020).** *An Elementary Introduction to Information Geometry*. Entropy 22(10):1100. Best modern compact reference.
- **Ay N., Jost J., Lê H. V., Schwachhöfer L. (2017).** *Information Geometry*. Ergebnisse vol 64. Springer. Coordinate-free formulation, Eguchi's theorem (divergence → metric + dual connections).
- **Pennec X., Sommer S., Fletcher T. (eds., 2020).** *Riemannian Geometric Statistics in Medical Image Analysis*. Elsevier. SPD + Fréchet means + Lie groups in practice.
- **Otto F. (2001).** The geometry of dissipative evolution equations: the porous medium equation. *Comm. PDE* 26:101-174. Founding paper for Wasserstein-as-Riemannian / W₂ gradient flow.

Sibling-package state already in repo (must not duplicate):
- `optim/transport/`: `Wasserstein1D`, `Sinkhorn`, `IQRNormalise`, `PairwiseWasserstein1D`. Doc.go:107-114 explicitly defers JKO / Wasserstein gradient flow / sliced-W / unbalanced / Gromov-W. ✓ This package owns OT primitives.
- `optim/proximal/`: FBS / FISTA / consensus ADMM with 8 prox operators. ✓ The proximal substrate JKO would compose against.
- `prob/`: `Distribution interface { PDF, CDF }` (no `LogPDF`, no `ScoreFunction`, no `FisherInfo`). Hook surface for closed-form Fisher.
- `linalg/`: Cholesky, LU, QR, eigen, PCA. ✓ Cholesky is the natural-gradient solve target.
- `chaos/`: RK4, Euler, Lorenz/VdP/Lyapunov. **No symplectic integrator.** Geodesic flow on non-trivial metrics needs Verlet/leapfrog.
- `geometry/`: quaternions, SDF, curves. Quaternion is *implicitly* exp/log on SO(3) but not exposed under that name.
- `autodiff/`: scalar reverse-mode tape (per 011/012/013). Forward-mode + Hessian (HVP) are absent — Christoffel-by-autodiff path is therefore blocked at v1.

Web research note: Geomstats' `information_geometry` submodule ships exactly: categorical/multinomial Fisher-Rao, Dirichlet, gamma, Poisson, Bernoulli, Gaussian (full + diagonal + univariate), with `metric_matrix`/`christoffels`/`exp`/`log`/`squared_dist`/`geodesic`. JAX-Cosmo (Campagne et al., 2023) is the cosmology Fisher-forecast use case (FIM via JVPs). These define the reasonable v1 scope.

---

### Tier 1 — must-ship for the package name to be honest

The 11 primitives below are the **minimum** that justifies the name `infogeo`. Each is small (≤200 LOC), has closed-form math, has an obvious cross-language golden-file plan, and unblocks ≥1 named consumer cite from `infogeo/doc.go`.

#### T1.1 — Fisher information matrix on the simplex (Fisher-Rao on multinomial)
**Math.** For categorical `p ∈ Δ^{n-1}` with `p_i > 0`, `g_{ij}(p) = δ_{ij}/p_i + 1/p_n` (sphere-pullback form via `√p_i`), or equivalently in the (n-1)-coord chart `g_{ij}(p) = δ_{ij}/p_i + 1/p_n`. **Diagonal in `√p` coordinates** — this is the cleanest impl.
**API.** `func FisherRaoSimplex(p []float64, out *linalg.Matrix) error`. Closed form. ~25 LOC. Singular at boundary `p_i = 0`; return `ErrSimplexBoundary`.
**Tier 1.1 because** every other simplex/categorical primitive (geodesic, exp, log, KL-distance ↔ Fisher quadratic) hangs off this.

#### T1.2 — Fisher information matrix for multivariate Gaussian
**Math.** `N(μ, Σ)` with parameter `θ = (μ, vech Σ)`:
`FIM_{μμ} = Σ⁻¹`, `FIM_{ΣΣ}` is the duplication-matrix Kronecker form `½ D_n^T (Σ⁻¹ ⊗ Σ⁻¹) D_n`, `FIM_{μΣ} = 0` (orthogonal blocks). Mardia-Kent-Bibby 1979 §3.4.
**API.** `func FisherRaoGaussian(mu []float64, sigma *linalg.Matrix, out *linalg.Matrix) error`. ~80 LOC + `linalg/Cholesky` reuse.
**Why.** 80% of practical IG calls. Standard ML / Heston / GARCH calibrators want this exact matrix.

#### T1.3 — generic Fisher via score function (autodiff path)
**Math.** `F(θ) = E_x[∂_θ log p(x;θ) ⊗ ∂_θ log p(x;θ)]`. Two paths:
(a) **Closed-form on exp-family** `p(x;θ) = h(x)·exp(η(θ)·T(x) - A(θ))`: `F = ∂²A/∂θ²` — single Hessian of log-partition. Zero MC error.
(b) **MC on arbitrary `Distribution`**: draw N samples, accumulate outer product of scores from `autodiff`. O(N·d²) memory unless we accumulate online.
**API.**
```go
type FisherInfoer interface { FisherInfo(out *linalg.Matrix) error }   // closed-form path
func FisherInfoMC(d prob.Distribution, theta []float64, n int, rng Rand, out *linalg.Matrix) error
```
**Blocker.** Closed-form path wants `prob/`'s `Distribution` interface to grow `LogPDF` + a `ScoreAt(x, θ, out []float64)` method (~30 LOC across Beta/Gamma/Normal/Cauchy). Autodiff fallback unblocked today modulo HVP gap (autodiff-missing T1).

#### T1.4 — Christoffel symbols (closed form for simplex + Gaussian)
**Math.** Levi-Civita: `Γ^k_{ij} = ½ g^{kl}(∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})`.
- Simplex (Fisher-Rao): `Γ^k_{ij} = -½ δ^k_i δ^k_j / p_k` (per Amari 2016 §2.4).
- Gaussian: closed form via Calvo-Oller 1991 / Skovgaard 1984 — Christoffel coefficients are products of `Σ⁻¹` entries, no autodiff needed.
**API.** `func ChristoffelSimplex(p []float64, out [][][]float64) error` and analogue for Gaussian.
**~40 LOC each.** Pinned against autodiff-Hessian-of-metric in tests once HVP lands.

#### T1.5 — α-connections ∇^α and α-divergence
**Math.** `∇^{(α)} = (1-α)/2·∇^{(m)} + (1+α)/2·∇^{(e)}`. α-divergence
`D_α(p∥q) = (1/(α(1-α))) · (1 - Σ p_i^α q_i^{1-α})` for α ∉ {0,1}; limits α=0 → KL(q∥p), α=1 → KL(p∥q), α=½ → 4·(1 - BC(p,q)) (Hellinger affinity form).
**API.** `func AlphaDivergence(p, q []float64, alpha float64) (float64, error)`. ~35 LOC.
**Numerical.** Same `p^α q^{1-α}` underflow hazard 091 §1.4 flagged for Rényi — fix once via `LogSumExp(α·log p_i + (1-α)·log q_i)` and reuse in both functions.
**Why Tier 1.** This single function is the bridge from f-divergence land (already shipped) to dual-connection land. Reduces "what's α=0 again?" friction.

#### T1.6 — Dual flat structure: e-coordinates ↔ m-coordinates conversion
**Math (exp-family).** `θ` = natural / e-coords (linear in η for the canonical exp-family); `η = ∇A(θ)` = expectation / m-coords; `θ ↔ η` via `A` and its Legendre dual `A*`. Dual flatness theorem (Amari 1985): the (e,m) coordinate pair is dually flat — both `∇^{(e)}` and `∇^{(m)}` have zero Christoffels in their respective coordinates.
**Concrete (multinomial).** e-coords `θ_i = log(p_i/p_n)`; m-coords `η_i = p_i`. Map both ways closed-form.
**Concrete (Gaussian).** e-coords `θ_1 = Σ⁻¹μ`, `θ_2 = -½Σ⁻¹`; m-coords `η_1 = μ`, `η_2 = Σ + μμ^T`. Closed-form invertible.
**API.** `Multinomial.ToETheta(p, out)`, `Multinomial.ToMTheta(p, out)`, `Gaussian.ToE…`/`ToM…`.
**~120 LOC across both manifolds.**
**Why Tier 1.** Without this, "dual flat" is a vibe; with it, e-geodesic and m-geodesic become straight lines in the right coords (next item).

#### T1.7 — e-geodesic and m-geodesic (mixture family + exponential family)
**Math.** In dual flat coords, geodesics are **literally straight lines** (no ODE):
- e-geodesic: `θ_t = (1-t)·θ_0 + t·θ_1` in e-coords; back-map to original chart.
- m-geodesic: `η_t = (1-t)·η_0 + t·η_1` in m-coords (i.e., for mixture family, the "mixture" of the two distributions: `p_t = (1-t)·p_0 + t·p_1`).
**API.** `EGeodesic(p0, p1, t)`, `MGeodesic(p0, p1, t)`. ~30 LOC each; the rest is the e/m coord map from T1.6.
**This is the v1 minimum geodesic** 091 §2.6 named: trivial numerically, no ODE, no symplectic integrator, no manifold projection — closes ~80% of practical geodesic-on-statistical-manifold use cases.

#### T1.8 — e-projection and m-projection
**Math.** Onto a submanifold M ⊂ statistical manifold:
- m-projection: `argmin_{q ∈ M} D_KL(p ∥ q)` — the maximum-likelihood projection (when M is exp-family in `q`).
- e-projection: `argmin_{q ∈ M} D_KL(q ∥ p)` — the I-projection / max-entropy projection (when M is m-flat).
**Pythagorean theorem (Csiszár 1975, Amari 1985).** If `r` is the m-proj of `p` onto an e-flat `M`, then `D_KL(p ∥ q) = D_KL(p ∥ r) + D_KL(r ∥ q)` for any `q ∈ M` — the IG analogue of orthogonal decomposition.
**API.** `MProject(p, manifold)`, `EProject(p, manifold)` — generic over a `Manifold` interface that ships only for exp-family submanifolds + mixture submanifolds at v1 (the two cases with closed-form solves).
**~150 LOC** (manifold interface + two concrete impls).
**Why Tier 1.** The single most-cited apparatus from the IG literature; the MoocIG / Nielsen tutorials all converge on this as the punchline.

#### T1.9 — Natural-gradient descent step
**Math.** `θ_{t+1} = θ_t - η · F(θ_t)⁻¹ · ∇L(θ_t)`. The step direction respecting the Fisher metric instead of Euclidean.
**API.**
```go
type NaturalGradientStepper struct{ Eta, Damping float64; FIM FisherInfoer }
func (s *NaturalGradientStepper) Step(theta, grad, out []float64) error
```
**Solve via Cholesky** of `F + λI` (Tikhonov-damped to handle boundary singularity 091 §2.4 flagged). **Conjugate-gradient variant** (Martens 2010 Hessian-Free) for large d via JVP — defer to v2.
**~80 LOC** (CG variant +120). Mirrors the existing `optim/` step-stepper pattern.

#### T1.10 — JKO scheme on the simplex (closed-form sub-case of W₂ gradient flow)
**Math.** Jordan-Kinderlehrer-Otto 1998: `ρ_{n+1} = argmin_ρ [F(ρ) + (1/2τ)·W₂²(ρ, ρ_n)]`. The discrete W₂ gradient flow.
**v1 scope.** Discrete simplex case — the inner W₂ becomes Sinkhorn (already in `optim/transport`); outer minimisation is convex if `F` is convex. ~60 LOC of glue against existing transport package.
**Why Tier 1.** Was named in `optim/transport/doc.go:108-110` as v2 deferred; this is the cross-package wire-up that lands it.
**Cross-package note.** Shape: lives in `infogeo/jko.go`, calls `transport.Sinkhorn` and `optim/proximal.FBS` — does not move math into `transport/` (which is OT-on-Euclidean), does not move math into `proximal/` (which is Euclidean prox). Composition lives here.

#### T1.11 — Stein discrepancy (kernel + log-density flavour)
**Math.** `KSD²(p, q) = E_{x,x' ~ q}[k_p(x, x')]`, where `k_p(x,x') = trace(∇_x ∇_{x'} k(x,x')) + ⟨∇log p(x), ∇_{x'} k(x,x')⟩ + ⟨∇log p(x'), ∇_x k(x,x')⟩ + ⟨∇log p(x), ∇log p(x')⟩·k(x,x')`. Liu-Lee-Jordan 2016, Chwialkowski-Strathmann-Gretton 2016.
**API.** `func KernelSteinDiscrepancy(samples [][]float64, scoreFn func(x []float64, out []float64), kernel KernelFn) (float64, error)`.
**~70 LOC.** Uses existing `mmd.go`'s kernel infrastructure — natural sibling of MMD (also a kernel U-statistic).
**Why Tier 1.** The modern goodness-of-fit substrate (since 2016). Pairs cleanly with the existing MMD; one extra kernel function (the Stein kernel) and one autodiff hook for the score, then existing MMD machinery.

---

### Tier 2 — fills out the textbook

12 primitives. Each is well-defined math but either (a) shipped as concrete instances of a Manifold interface that doesn't exist yet, or (b) pulls a sibling package change first.

#### T2.1 — `Manifold` / `RiemannianMetric` interface
The Geomstats-shaped contract. Without it, every primitive below is a bespoke function. Recommended shape (see 094-infogeo-api):
```go
type Manifold interface {
    Dim() int
    Project(point []float64, out []float64) error           // retraction onto manifold
    InnerProduct(point, u, v []float64) float64
}
type RiemannianMetric interface {
    Manifold
    Exp(point, tangent []float64, out []float64) error
    Log(point, target []float64, out []float64) error
    GeodesicDistance(p, q []float64) float64
    ParallelTransport(point, tangent, direction []float64, out []float64) error
}
```
**~80 LOC** for the interface + adapter scaffolding. Every Tier 2 item below implements one or both.

#### T2.2 — exponential map / log map (per-manifold closed forms)
**Simplex (Fisher-Rao via sphere embedding).** `Exp_p(v)_i = (√p_i · cos(‖v‖_p) + (v_i / 2‖v‖_p) · sin(‖v‖_p))²`. Use `cos(π-x) = -cos(x)` for `‖v‖>π/2` per 091 §2.3. ~40 LOC.
**Gaussian (Fisher-Rao).** No fully-closed form for the full (μ, Σ) FIM; **Calvo-Oller 1991** maps to upper-triangular and gives a one-step exp via SPD. Reuse SPD code (T2.3).
**Sphere S^{n-1}.** `Exp_p(v) = cos‖v‖·p + sin‖v‖/‖v‖·v`. ~15 LOC.
**Hyperbolic (Lorentz model).** `Exp_p(v) = cosh‖v‖·p + sinh‖v‖/‖v‖·v` with the Minkowski inner product. ~25 LOC.
**Hyperbolic (Poincaré ball).** Möbius addition formulation, Ungar 2008. ~50 LOC.

#### T2.3 — SPD manifold (symmetric positive definite matrices)
**Affine-invariant metric** (Pennec-Fillard-Ayache 2006). `g_P(X, Y) = trace(P⁻¹·X·P⁻¹·Y)`.
**Exp:** `Exp_P(X) = P^½ · expm(P^{-½}·X·P^{-½}) · P^½`. Needs matrix exponential (not in `linalg/` — sub-blocker).
**Log:** symmetric. Needs matrix log.
**Distance:** `d(P,Q) = ‖logm(P^{-½}·Q·P^{-½})‖_F`.
**Why this matters.** Diffusion-tensor imaging, Riemannian PCA, multivariate Gaussian ↔ SPD bijection (Σ ↔ Cholesky factor). Pennec-Sommer-Fletcher 2020 §3.
**Sub-blocker.** Needs `linalg.MatrixExp` and `linalg.MatrixLog` (Higham 2008 squaring-and-scaling — ~150 LOC each, candidate for `linalg/special.go`). Spawns a separate ticket.

#### T2.4 — sphere S^{n-1} as manifold
Closed-form everything: exp, log, parallel transport via parallel-transport-along-geodesic, geodesic distance = arccos⟨p,q⟩, retraction = renormalise. ~80 LOC. The "hello world" manifold; should land first to validate the `Manifold` interface shape.

#### T2.5 — hyperbolic spaces (Lorentz + Poincaré ball)
Both isometric models. Lorentz preferred numerically (no ‖θ‖<1 boundary singularity); Poincaré preferred for visualisation. Ship both with a closed-form bijection between them. ~150 LOC. Cite Ungar 2008 (gyrovector approach) + Nickel-Kiela 2017 (Poincaré embeddings).

#### T2.6 — Stiefel manifold V(n,k) and Grassmann manifold Gr(n,k)
**Stiefel:** orthonormal k-frames in R^n. `V(n,k) = {X ∈ R^{n×k} : X^T X = I_k}`. Edelman-Arias-Smith 1998 closed-form exp via QR.
**Grassmann:** k-planes in R^n = V(n,k) / O(k) — quotient. Exp/log via SVD.
**Why.** PCA on subspaces, density estimation on Stiefel/Grassmann (per topic prompt). Pymanopt's bread-and-butter.
**~250 LOC combined.** Both use `linalg.QR` / `linalg.SVD` — both already in repo.

#### T2.7 — parallel transport
Closed form on each Tier 2 manifold (sphere: Schild's-ladder closed; hyperbolic: closed; SPD: needs matrix sqrt; Stiefel/Grassmann: Edelman 1998).
**Generic numerical fallback:** Schild's ladder + 4-step refinement (Lorenzi-Pennec 2014). ~100 LOC for the generic, ~30 LOC per manifold for closed-form.

#### T2.8 — geodesic ODE solver + geodesic shooting (boundary value)
**Initial-value (`Exp`):** Hamilton's equations `dq/dt = ∂H/∂p, dp/dt = -∂H/∂q` on `H = ½g^{ij}p_ip_j`. Symplectic Verlet / leapfrog. ~80 LOC.
**Cross-package blocker.** `chaos/` has RK4 only — geodesic ODE drifts off the manifold under RK4 per 091 §2.2. Needs `chaos.Verlet` (~80 LOC) shipped first.
**Boundary-value (`Log` from far away):** geodesic shooting via Newton on `Exp_p(v) - q = 0`; uses `optim/` Newton or L-BFGS. ~60 LOC.

#### T2.9 — curvature tensor (Riemann + Ricci + scalar)
`R^l_{ijk} = ∂_i Γ^l_{jk} - ∂_j Γ^l_{ik} + Γ^l_{im}Γ^m_{jk} - Γ^l_{jm}Γ^m_{ik}`. Then Ricci `R_{ij} = R^k_{ikj}` and scalar `R = g^{ij}R_{ij}`.
**Closed-form on Fisher-Rao simplex:** sectional curvature is constant `+1/4` (sphere of radius 2 in `√p` coords). One-line answer.
**Closed-form on Gaussian:** Skovgaard 1984; mixed signs.
**Generic numerical:** autodiff of Christoffels. Needs HVP (autodiff-missing T1).
**~120 LOC** for the closed-form pair + a stub for the generic.

#### T2.10 — Mirror descent
**Math.** `θ_{t+1} = ∇φ*(∇φ(θ_t) - η ∇L(θ_t))`. Beck-Teboulle 2003. With `φ = -entropy`, this is exactly EG / multiplicative-weights on the simplex; with `φ = ½‖·‖²` it reduces to gradient descent.
**API.** Composes the existing `Bregman` infrastructure: caller supplies `(phi, gradPhi, gradPhiStar)`. ~40 LOC.
**Why.** "Natural" optimiser on the simplex. Same numerical content as natural-gradient when φ is the FIM-derived divergence.

#### T2.11 — EM in IG-form (alternating e-projection / m-projection)
**Math (Amari-Kurata 1995, Csiszár-Tusnády 1984).** EM for latent-variable models = alternating projections between an m-flat manifold (data submanifold) and an e-flat manifold (model submanifold). Each step is one m-projection or one e-projection.
**API.** `func AlternatingProjection(model, data Manifold, p0 []float64, maxIter int) ([]float64, error)`. ~80 LOC over T1.8.
**Why Tier 2.** Re-derives EM from IG primitives — the 2-page pedagogical demonstration that justifies the package's existence to anyone who's seen EM before.

#### T2.12 — Stein variational gradient descent (SVGD)
**Math.** Liu-Wang 2016. Particle-based variational inference: `x_i ← x_i + η · φ(x_i)` with `φ(x) = (1/N) Σ_j [k(x_j, x) ∇log p(x_j) + ∇_{x_j} k(x_j, x)]`. Newton-style on Wasserstein-Stein geometry.
**API.** `func SVGD(particles [][]float64, scoreFn ScoreFunc, kernel KernelFn, eta float64, steps int) error`. ~80 LOC.
**Why Tier 2.** The default modern Bayesian-inference primitive in 2026 (since the variational-inference renaissance ~2018). Uses Stein discrepancy (T1.11) + autodiff scores.

---

### Tier 3 — research-frontier and specialist

13 primitives. Each is real and citable but either (a) niche, (b) requires research-grade numerical care, or (c) only justifies its weight on a v3+ proof-of-concept.

#### T3.1 — Wasserstein gradient flow (continuous)
JKO is the discrete-time scheme (T1.10). The continuous-time flow `∂ρ/∂t = ∇·(ρ ∇(δF/δρ))` for free energy `F` is a PDE. Otto 2001 / Ambrosio-Gigli-Savaré 2008. Ship as a stepper class with user-supplied `δF/δρ` functional gradient. ~150 LOC.

#### T3.2 — sliced Wasserstein on Stiefel/Grassmann
Sliced-W² over random projections to lines (Kolouri-Pope-Martin-Rohde 2018) lifted to Stiefel/Grassmann projections. Naturally lives in `optim/transport/` not `infogeo/`.

#### T3.3 — Hessian-free / conjugate-gradient natural gradient
Martens 2010, K-FAC (Martens-Grosse 2015). `F·v` matvec via JVP, no explicit FIM materialisation. Required when `dim(θ) > ~1000`. ~150 LOC + needs forward-mode AD (autodiff missing).

#### T3.4 — Fréchet mean on a manifold
`argmin_p (1/N) Σ d²(p, p_i)`. Karcher 1977. Closed-form on simplex via centroid in `√p` coords; iterative on SPD/Stiefel. ~80 LOC each.

#### T3.5 — Karcher / Fréchet variance
`(1/N) Σ d²(p_i, μ)` — the Riemannian counterpart of variance. Trivial once T3.4 lands.

#### T3.6 — Riemannian PCA / tangent PCA
Project samples to the tangent space at the Fréchet mean, run PCA, lift back via Exp. Fletcher-Lu-Pizer-Joshi 2004. ~100 LOC; reuses `linalg.PCA` + Tier 2 manifolds.

#### T3.7 — α-projection (general α, not just e/m)
α-flat submanifold projection — generalises T1.8 to interpolation between e- and m-projection. Niche.

#### T3.8 — Bures-Wasserstein metric on SPD
The OT metric on Gaussian distributions parameterised by their covariance: `d²_BW(P, Q) = trace(P + Q - 2(P^½ Q P^½)^½)`. Bhatia-Jain-Lim 2019. Different metric on SPD than the affine-invariant one (T2.3); ship both, document when to use which.

#### T3.9 — second-order natural gradient (Hessian-corrected)
`(F + ½∇²L) δ = ∇L`. The Newton-direction analogue. Needs Hessian (autodiff missing).

#### T3.10 — divergence-induced metric and dual connections (Eguchi)
**Eguchi 1992.** From any divergence `D(p∥q)`, derive: metric `g_{ij}(p) = -∂_i ∂_j' D|_{p=q}`, primal connection `Γ_{ijk}(p) = -∂_i ∂_j ∂_k' D`, dual `Γ*_{ijk}(p) = -∂_i' ∂_j' ∂_k D`. Recovers Fisher metric + α-connections from α-divergence. ~60 LOC; the theoretical bridge.

#### T3.11 — entropic Wasserstein (debiased Sinkhorn divergence)
Feydy-Séjourné-Vialard-Amari-Trouvé-Peyré 2019. Already noted as deferred in `optim/transport/doc.go:99-101`. Ships in `transport/`, not here.

#### T3.12 — Information-geometric MCMC (Riemannian-manifold HMC)
Girolami-Calderhead 2011. RMHMC uses the local Fisher metric to set the HMC mass matrix. ~200 LOC; pairs naturally with autodiff and FIM. Niche but high-leverage for Bayesian users.

#### T3.13 — α-NMF / Tsallis-flavoured non-negative matrix factorisation
Cichocki-Cruces-Amari 2011. α-divergence-cost NMF — niche but a clean demo of why α-divergences are useful (interpolates between Frobenius / KL-NMF / IS-NMF objectives).

---

### Cross-package coordination ledger

Each item below blocks ≥1 Tier 1 / Tier 2 primitive above. Order of operations is:

| Blocker | Lives in | Blocks | LOC | Notes |
|---|---|---|---|---|
| `chaos.Verlet` (symplectic integrator) | `chaos/` | T2.8 (geodesic ODE) | ~80 | per 091 §2.2 |
| `linalg.MatrixExp` / `linalg.MatrixLog` | `linalg/` | T2.3 (SPD) | ~300 | Higham 2008 |
| forward-mode AD + HVP | `autodiff/` | T1.4 generic, T2.9 | ~600 | per autodiff-missing T1 (012) |
| `prob.Distribution.LogPDF` + `ScoreAt` | `prob/` | T1.3 closed form | ~30 + per-distribution | the `Distribution` interface grows |

The Tier 1 set above can ship **without** any of these blockers if scoped to (a) closed-form simplex/Gaussian Fisher (no autodiff path), (b) e/m geodesics (no ODE), and (c) closed-form e/m projections only. That is the lean v1: ~700 LOC of pure additions, zero blockers, no sibling-package changes.

---

### Sprint-ordered recommendations

**Sprint A (lean v1, no blockers, ~700 LOC):**
1. T1.5 `AlphaDivergence` + LSE-based inner-loop (also fixes 091 R3 Rényi underflow). 35 LOC.
2. T1.1 `FisherRaoSimplex`. 25 LOC.
3. T1.6 multinomial e/m coord conversion. 60 LOC.
4. T1.7 e/m geodesics on simplex. 40 LOC.
5. T1.8 e/m-projection on simplex (closed-form for exp-family submanifolds). 100 LOC.
6. T1.9 natural-gradient stepper (using simplex FIM + `linalg.Cholesky`). 80 LOC.
7. T1.2 + T1.4 + T1.6/T1.7/T1.8 Gaussian instances. 250 LOC.
8. T1.11 Kernel Stein Discrepancy (composes against existing `mmd.go`). 70 LOC.

**Sprint B (interface lift + sibling-package wiring, ~1,200 LOC):**
9. T2.1 `Manifold` / `RiemannianMetric` interface — the API design call paired with 094.
10. T2.4 sphere S^{n-1}, T2.5 hyperbolic — first non-statistical manifolds.
11. T1.10 JKO scheme — wires `transport.Sinkhorn` + `proximal.FBS`.
12. T2.10 mirror descent — composes existing `Bregman` machinery.
13. T2.11 IG-form EM, T2.12 SVGD.

**Sprint C (cross-package blockers + research frontier):**
14. `chaos.Verlet` → T2.8 geodesic ODE.
15. `linalg.MatrixExp/Log` → T2.3 SPD.
16. autodiff HVP → T1.3 generic Fisher, T1.4 generic Christoffel.
17. T2.6 Stiefel/Grassmann (after the SPD machinery validates the matrix-manifold pattern).
18. T3.* tier as use cases pull.

**Single highest-leverage item.** T1.6 + T1.7 + T1.8 *as a triple* (e/m coords ↔ geodesics ↔ projections on the simplex). Closed form, no ODE, no blocker, ~200 LOC, and after they land the package's name is *finally* honest: it ships **information geometry on the multinomial manifold**. Every later Tier 1/Tier 2 expansion is a structural copy-paste of these three primitives onto another manifold. Without them, no narrative; with them, a clear roadmap.

---

End of audit. 092 / 400.

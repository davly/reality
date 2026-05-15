# 226 | new-hyperbolic-embed

**Summary (2 lines):** reality v0.10.0 ships ZERO hyperbolic-geometry surface — repo-wide grep on `Hyperbol|Poincar|Lorentz|Mobius|gyrovector|arccosh|MobiusAdd|Klein|hyperboloid` across all `*.go` files returns only `gyro` (gyroscope sensor in `control/filter.go::ComplementaryFilter`) and `cosh/sinh/tanh` as transcendental sites in `autodiff/ops.go` (Euclidean scalar-tape backward rules) + `audio/separation/ica.go` (ICA contrast); `geometry/` ships quaternion + SDF + curves only (no negative-curvature manifold), `infogeo/` ships f-divergences + Bregman + MMD only and explicitly defers hyperbolic to v2 per `092-infogeo-missing.md` T2.5 (~150 LOC scoped), `optim/` ships Euclidean GD/L-BFGS/IP/GA/SA/proximal/transport only with no Riemannian retraction surface (per 206-T0). This slot scopes the hyperbolic-embedding canon as twenty-three primitives H1–H23 totalling ~3,400 LOC under a new sub-package `geometry/hyperbolic/` (mirroring the `optim/proximal/` + `optim/transport/` precedent), with **the Lorentz + Poincaré ball + closed-form bijection triple (~360 LOC) as the Tier-1 keystone** validating the negative-curvature contract on the two industry-standard models with cross-language golden-file parity at 1e-12 round-trip and 1e-12 model-bijection-isometry. **Disambiguation versus 206-new-riemannian-opt-R6** (Hyperbolic as one of seven `Manifold` instances under `optim/manifold/hyperbolic.go`, ~250 LOC): 206-R6 scopes hyperbolic only as an *optimisation domain* (Manifold interface methods Exp/Log/Retract/PT). This slot 226 is the *complete hyperbolic-geometry canon*, including all four classical models (Poincaré disk/ball, Lorentz/hyperboloid, Klein, upper-half-space), Möbius transformations, gyrovector arithmetic (Ungar 2008), Sarkar's tree-embedding, distortion bounds, hyperbolic Voronoi, hyperbolic NN primitives (Ganea-Bécigneul-Hofmann 2018), Poincaré embeddings (Nickel-Kiela 2017), mixed-curvature products. Recommended: ship geometry canon ONCE in `geometry/hyperbolic/`; `optim/manifold/hyperbolic.go` (206-R6) is a 30-LOC `Manifold` wrapper. **Disambiguation versus 207-new-diff-geo-D9** (generic geodesic-ODE on intrinsic `g_ij`, cites hyperbolic-plane half-circle as a validation): 207-D9 ships generic substrate, 226 H1–H4 ships closed-form (`Exp_x(v) = cosh(‖v‖_L)·x + sinh(‖v‖_L)/‖v‖_L · v` on Lorentz), 5–8× faster and exact to round-off; cross-language parity ships unblocked here without 207's chaos/symplectic blocker.

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Verified by direct read.

- **Hyperbolic-canon grep**: `Hyperbol|Poincar|Lorentz|Mobius|gyrovector|arccosh|Klein|hyperboloid` across all `*.go` returns *zero* hits in any source file (including tests). Only `gyro` matches as gyroscope (control/filter.go) and `cosh/sinh/tanh` as Euclidean autodiff/ICA non-linearities.
- **`geometry/`** (~720 LOC): `quaternion.go` ships S³ rotation (constant *positive* curvature — the dual of hyperbolic). Negative-curvature analogue is structurally absent.
- **`infogeo/`** (~1,373 LOC): `bregman.go` ships squared-Eucl/genKL/IS only. Hyperbolic Bregman generator `φ(x) = ⟨x, log x⟩ − x_0 log x_0` on Lorentz cone (Bauschke-Borwein 2001 §6) absent. `doc.go:55-79` defers hyperbolic to v2 per 092-T2.5.
- **`optim/`** (~1,800 LOC): zero `Manifold` interface, zero retraction primitives. `proximal.ProxL2Ball` is Euclidean ball indicator-prox, NOT Poincaré-ball-as-manifold projection (which is `x ← x · min(1, (1−ε)/‖x‖)` Euclidean clipping inside ‖x‖<1).
- **`graph/`** (~1,200 LOC): zero hyperbolic graph embedding (cross-link to H10/H12); zero hyperbolic Voronoi (cross-link to H14).
- **Cross-link audit**:
  - **092-T2.5** (hyperbolic, 150 LOC, Ungar 2008 + Nickel-Kiela 2017): closest review precedent, punted to v2; this slot 226 IS the implementation T2.5 deferred.
  - **206-R6** (250 LOC `Manifold` instance): 206 wraps this slot's H1+H2+H8 with Manifold contract — ~30-LOC wrapper, no duplication.
  - **207-D9**: generic geodesic-ODE substrate; 226 ships closed-form Lorentz Exp/Log, 5–8× faster, no ODE drift.
  - **177-synergy / 205-Lie-L6 / 107-graph-missing / calculus**: no overlap; 226 fills the hyperbolic-graph-embedding gap that 107 declined.

---

## 1. The twenty-three primitives

For each: **(a) ships**, **(b) add**, **(c) LOC**, **(d) blocker**.

### H1 — Lorentz / hyperboloid model H^n ⊂ ℝ^{n+1}
(a) Nothing. (b) `H^n = {x ∈ ℝ^{n+1} : ⟨x,x⟩_L = −1, x_0 > 0}` with Minkowski inner `⟨u,v⟩_L = −u_0v_0 + Σu_iv_i` (signature `(−,+,…,+)`). Ships `MinkowskiInner`, `IsOnHyperboloid`, `LorentzProject` (renormalise via `x/√(−⟨x,x⟩_L)`), tangent space `T_xH^n = {v : ⟨x,v⟩_L = 0}`, `LorentzProjectTangent(v) = v + ⟨x,v⟩_L·x` (note: PLUS because `⟨x,x⟩_L = −1`), `LorentzNorm(v) = √⟨v,v⟩_L`. Ref: Ratcliffe 2019 *Foundations of Hyperbolic Manifolds* §3.1; Cannon-Floyd-Kenyon-Parry 1997 *Flavors of Geometry* §2; Pennec-Sommer-Fletcher 2020 §2.1. (c) ~80 LOC `geometry/hyperbolic/lorentz.go`. (d) None.

### H2 — Lorentz Exp / Log / Distance closed-form
(a) Nothing. (b) `Exp_x(v) = cosh(‖v‖_L)·x + sinh(‖v‖_L)/‖v‖_L · v` with `‖v‖→0` Taylor `sinh(t)/t = 1 + t²/6 + O(t⁴)` switched at `‖v‖_L < 6e-8`; `Log_x(y) = θ·(y + cosh(θ)·x)/sinh(θ)` with `θ = arccosh(−⟨x,y⟩_L)`, near-1 stability via `arccosh(1+u) = log(1 + u + √(u(2+u)))`; `d_H(x,y) = arccosh(−⟨x,y⟩_L)`; retraction `Retract_x(v) = LorentzProject(x+v)`. Ref: Pennec 2020 §2.1; Skopek-Ganea-Bécigneul ICLR 2020 §3.2. (c) ~80 LOC same file. (d) None.

### H3 — Poincaré ball B^n ⊂ ℝ^n
(a) Nothing. (b) `B^n = {x ∈ ℝ^n : ‖x‖<1}` with conformal metric `g_x = (2/(1−‖x‖²))²·I_n`. Ships `IsInPoincareBall`, `PoincareProject(x; eps=1e-5)` via `x · min(1, (1−eps)/‖x‖)` (Nickel-Kiela 2017 §3 standard projection — prevents 1/(1−‖x‖²) singularity), `d_B(x,y) = arccosh(1 + 2‖x−y‖²/((1−‖x‖²)(1−‖y‖²)))`, conformal factor `λ_x = 2/(1−‖x‖²)`, `PoincareInner(u,v) = λ_x²·⟨u,v⟩_E`. Tangent space `T_xB^n = ℝ^n` (open ball; tangent at every point is full ℝ^n with metric `g_x`). Ref: Cannon-FKP 1997 §2.4; Ratcliffe 2019 §4.6. (c) ~80 LOC `geometry/hyperbolic/poincare.go`. (d) None.

### H4 — Möbius arithmetic + scalar mul + matvec (gyrovector)
(a) Nothing. (b) Ungar 2008 gyrovector formalism: Möbius addition `x⊕y = ((1+2⟨x,y⟩+‖y‖²)x + (1−‖x‖²)y) / (1+2⟨x,y⟩+‖x‖²‖y‖²)` (non-commutative, non-associative, `0⊕y = y`); subtraction `x⊖y = x⊕(−y)`; scalar mul `r⊗x = tanh(r·arctanh(‖x‖))·x/‖x‖`; gyration `gyr[a,b]·c = ⊖(a⊕b)⊕(a⊕(b⊕c))` measures non-associativity. **Möbius matvec for hyperbolic NN** (Ganea-Bécigneul-Hofmann 2018 §3): `M⊗x = tanh(‖M·x_E‖·arctanh(‖x‖)/‖x‖)·M·x_E/‖M·x_E‖`. Exp/Log via gyrovector: `Exp_0(v) = tanh(‖v‖)·v/‖v‖`, `Log_0(y) = arctanh(‖y‖)·y/‖y‖`, `Exp_x(v) = x⊕Exp_0(λ_x·v/2)`, `Log_x(y) = (2/λ_x)·Log_0(⊖x⊕y)`. Ref: Ungar 2008 *Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity* §3 (canonical); Ungar 2014 §3. (c) ~140 LOC `geometry/hyperbolic/mobius.go`. (d) None.

### H5 — Klein / Beltrami-Klein model K^n
(a) Nothing. (b) Same set as Poincaré ball but projective metric `g_x^K = (1−‖x‖²)^{-1}·I + (1−‖x‖²)^{-2}·xxᵀ`. **Geodesics on K^n are straight Euclidean chords** — non-conformal (angles distorted) but visually faithful for tree-structure. `d_K(x,y) = arccosh((1−⟨x,y⟩)/√((1−‖x‖²)(1−‖y‖²)))`. Bijection from Lorentz: `K(x) = (x_1,…,x_n)/x_0` (dehomogenisation). Use case: hyperbolic Voronoi (H14) — bisectors are Euclidean half-planes (Boguñá-Krioukov-Almagro-Serrano 2021 *Nature Reviews Physics* 3:114 §3.1). (c) ~80 LOC `geometry/hyperbolic/klein.go`. (d) None.

### H6 — Upper half-space U^n
(a) Nothing. (b) `U^n = {x ∈ ℝ^n : x_n > 0}`, metric `g_x = (1/x_n²)·I_n`. Geodesics: vertical lines or semicircles orthogonal to boundary. Conformal. n=2: `d_U(z,w) = arccosh(1 + |z−w|²/(2·Im(z)·Im(w)))`. Bijection to Poincaré (n=2 Cayley transform `z↦(z−i)/(z+i)`; n-D inversion). Use case: number theory (`SL(2,ℤ)\U²` modular forms), Teichmüller; less common in ML. Ref: Ratcliffe 2019 §4.6+§6.2; Beardon 1983 *Geometry of Discrete Groups* §7. (c) ~80 LOC. (d) None.

### H7 — Closed-form bijections between models
(a) Nothing. (b) Four canonical bijections (8 functions), each an isometry: **Lorentz↔Poincaré** (stereographic projection from `(−1,0,…,0)`): `L→P: y_i = x_i/(1+x_0)`; `P→L: x_0 = (1+‖y‖²)/(1−‖y‖²), x_i = 2y_i/(1−‖y‖²)`. Most-used. **Lorentz↔Klein** (dehomogenisation, see H5). **Poincaré↔Klein** and **Poincaré↔Half-space** (Cayley, see H6). Cross-validate: `d_M1(P_M1(x), P_M1(y)) = d_M2(x,y)` to 1e-12 — *the cross-model isometry test catches sign errors.* Ref: Cannon-FKP 1997 §2.5 (model-equivalence theorem); Ratcliffe 2019 §6.3. (c) ~120 LOC `geometry/hyperbolic/bijection.go`. (d) None.

### H8 — Parallel transport (closed-form)
(a) Nothing. (b) Lorentz: `PT_{x→y}(v) = v − ⟨Log_x(y), v⟩_x/d²(x,y) · (Log_x(y) + Log_y(x))`. Poincaré gyration formula: `PT_{x→y}(v) = gyr[y, ⊖x] · ((1−‖x‖²)/(1−‖y‖²)) · v`. Both pull back to same operator under H7 bijection. Closed-form preferred over Schild's-ladder generic (`O(step²)` per rung). Ref: Lorenzi-Pennec 2014 *IJCV* 105:111; Ganea 2018 §3.2 (gyration). (c) ~70 LOC. Pairs with 206-R20 + 207-D14. (d) None.

### H9 — Möbius transformations / hyperbolic isometries
(a) Nothing. `quaternion.go` ships SO(3) (positive-curvature S² isometries); negative-curvature analogue absent. (b) Isometry group `O^+(n,1) = SO^+(n,1)` (Lorentz transformations preserving future cone) acting on `H^n`; Möbius transformations on Poincaré ball. Lorentz boost `Λ(v)` (cross-link to special relativity — same matrices as relativistic boosts in hypothetical 174-relativity scope), spatial-rotation lift `diag(1, R)` for `R∈SO(n)`, 2D Möbius `z↦(az+b)/(cz+d)` with `ad−bc=1` (PSL(2,ℝ) acts on U², PSL(2,ℂ) on H³), n-D Möbius via `O^+(n,1)↔Möbius(S^{n−1})` boundary action. Composition: relativistic velocity-addition; non-commutativity of boosts produces Wigner rotation (= H4 gyration). Ref: Ratcliffe 2019 §3.2; Beardon 1983 §3; MTW 1973 §2. (c) ~120 LOC `geometry/hyperbolic/isometry.go`. (d) None; cross-link only to 174-relativity (single source-of-truth here).

### H10 — Sarkar's tree-embedding algorithm
(a) Nothing. (b) Sarkar 2011 *Graph Drawing*: embeds tree with `n` nodes into Poincaré disk B² with `(1+ε)` metric distortion for *any* ε > 0, by recursive embedding with scaling `τ ≥ 1+log(deg_max)/ε`. Algorithm: place root at origin; for each node, Möbius-translate to origin, rotate parent-axis to negative real, place children evenly-spaced at distance τ, Möbius-translate back. Sala-De Sa-Gu-Ré ICLR 2018 §4 multidim generalisation: precision-vs-dimension tradeoff `1+O(2^{-d/log(b)})`. Ref: Sarkar 2011; Sala 2018. (c) ~180 LOC `geometry/hyperbolic/sarkar.go`. Consumes H4. (d) None; reads `graph.Graph` with `IsTree()` precondition.

### H11 — Distortion bounds
(a) Nothing. (b) Canonical theoretical bounds as runtime functions: **Linial-London-Rabinovich 1995** *Combinatorica* 15:215 — any n-point metric embeds Euclideanly with `Θ(log n)` distortion (lower bound); trees thus `Ω(log n)` distortion in Euclidean — *the motivation for hyperbolic*. **Bartal 1996** *FOCS* — probabilistic-tree-metric upper bound. **Sarkar 2011** §3 — `(1+ε)` upper bound on Poincaré disk (constant-curvature gives arbitrary-low-distortion). **Sala 2018** §3 — multidim precision bound. **Krioukov-Papadopoulos-Vahdat 2010** *PRE* 82:036106 — `b^r`-vs-`r^d` cardinal-difference theorem. Ship `MeasureDistortion(d_G, embedding f, samples) → (worst, average)`. Ref: LLR 1995; Bartal 1996; Sarkar 2011; Sala 2018; Krioukov 2010. (c) ~230 LOC `geometry/hyperbolic/distortion.go`. (d) None.

### H12 — Nickel-Kiela 2017 graph embedding
(a) Nothing. (b) Nickel-Kiela 2017 *NeurIPS Poincaré Embeddings for Learning Hierarchical Representations*: gradient-based learning of Poincaré-ball embeddings for hierarchical graphs. Riemannian-SGD update `grad^R = (1−‖x‖²)²/4 · grad^Eucl` (the inverse-Riemannian-metric scaling); negative-sampling soft-max loss `L = −log(exp(−d(u,v))/Σ_n exp(−d(u,n)))`; project after step `x ← PoincareProject(x − η·grad^R, eps)`. Pairs with 206-R13 (R-SGD canon) — Nickel-Kiela's update IS RSGD on Poincaré with this specific loss. Ref: Nickel-Kiela 2017 §3. (c) ~220 LOC `geometry/hyperbolic/embed_graph.go`. (d) Soft cross-link to 206-R13 (this slot ships Poincaré-specific RSGD loop unblocked).

### H13 — Tifrea 2019 Poincaré-GloVe word embedding
(a) Nothing. (b) Tifrea-Bécigneul-Ganea ICLR 2019 *Poincaré GloVe: Hyperbolic Word Embeddings*: loss `L = Σ_{i,j} f(X_ij)·(d_H(u_i, v_j) + b_i + c_j − log X_ij)²` where `X_ij` is co-occurrence, `f` GloVe weighting, `d_H` hyperbolic distance. Outperforms Euclidean GloVe on hierarchical relations (hypernymy); matches on flat. Same RSGD update as H12. Ref: Tifrea 2019. (c) ~150 LOC `geometry/hyperbolic/embed_word.go`. (d) None.

### H14 — Hierarchical clustering via hyperbolic Voronoi
(a) Nothing. `graph/community.go` ships Euclidean only. (b) Boguñá-Krioukov-Almagro-Serrano 2021 *Nature Reviews Physics* 3:114: hyperbolic Voronoi cells have *exponentially* growing volumes near boundary — natural geometry of hierarchical data. In Klein model bisectors are Euclidean half-planes (Boguñá §3.1) — reduces to Euclidean Voronoi in Klein coordinates, back-transform via H5. (c) ~180 LOC `geometry/hyperbolic/voronoi.go`. (d) **Sub-blocker**: Euclidean Voronoi primitive (~250 LOC, separate slot or `geometry/voronoi.go` infill).

### H15 — Hyperbolic neural network primitives
(a) Nothing. (b) Ganea-Bécigneul-Hofmann NeurIPS 2018 *Hyperbolic Neural Networks*: hyperbolic linear `f(x) = M⊗x ⊕ b` (H4 Möbius matvec + add); hyperbolic activation `σ_hyp(x) = Exp_0(σ(Log_0(x)))` (lift, apply, push); hyperbolic softmax / multinomial logistic regression with distance-to-decision-hyperplane scoring; hyperbolic FFN (composition); hyperbolic attention (Gulcehre et al. ICLR 2019 *Hyperbolic Attention Networks*). Ref: Ganea 2018 §3-§4; Gulcehre 2019. (c) ~260 LOC `geometry/hyperbolic/nn.go`. Consumes H3+H4+H7+H8. (d) None.

### H16 — Mixed-curvature embedding
(a) Nothing. (b) Skopek-Ganea-Bécigneul ICLR 2020 *Mixed-Curvature VAE*: product manifold `M = M_1×…×M_k` with each `M_i` constant-curvature `κ_i` (hyperbolic, flat, spherical mixable); composed Exp/Log/PT; learnable curvatures. Distance product `d_M² = Σ d_{M_i}²`. Use case: hierarchical-with-cyclic-structure data. Ref: Skopek 2020. (c) ~150 LOC `geometry/hyperbolic/mixed.go`. (d) Soft cross-link to 206-R2 (Sphere) for `κ>0` component.

### H17 — Hyperbolic PCA (tangent-space PCA at Karcher mean)
(a) Nothing. (b) Fletcher-Lu-Pizer-Joshi 2004 *IEEE TMI* 23:995 generic tangent-PCA instantiated on H^n: Karcher μ via H21, project samples to `T_μH^n` via Log, run linear PCA, lift via Exp. Pairs with 206-R16. Ref: Fletcher 2004; Pennec 2018. (c) ~50 LOC mostly tests. (d) None beyond 206-R16.

### H18 — Klein chord-path visualisation
(a) Nothing. (b) `KleinChordPath(p1, p2 Vec2, nSamples) → []Vec2` returning samples along Klein-coordinate Euclidean chord (geodesic). Plotting itself out-of-scope per CLAUDE.md "library not service". (c) ~30 LOC `geometry/hyperbolic/visualise.go`. (d) None.

### H19 — Hyperbolic random sampling
(a) Nothing. (b) Uniform-on-ball-of-radius-r (radial CDF inversion, hyperbolic ball volume `V_r = (n−1)·∫_0^r sinh^{n−1}(t)dt`); wrapped-normal (Mathieu et al. ICLR 2019 *Continuous Hierarchical Representations with Poincaré VAE*); Riemannian-Gaussian `p(x) ∝ exp(−d²(x,μ)/(2σ²))` with closed-form 2D normaliser, numerical higher-D (Said-Hajri-Bombrun-Vemuri 2017). Ref: Mathieu 2019; Nagano-Yamaguchi-Fujita-Koyama ICML 2019 *A Wrapped Normal Distribution on Hyperbolic Space*. (c) ~150 LOC `geometry/hyperbolic/sample.go`. (d) None.

### H20 — Information geometry of hyperbolic distributions
(a) Nothing. (b) Hyperbolic divergence `D_H(P||Q) = −(⟨P,Q⟩_L + 1)` (Bauschke-Borwein 2001 §6 — proportional to `cosh(d_H)−1`); Bregman generator on Lorentz cone `φ(x) = ⟨x, log x⟩ − x_0·log x_0`; Bures-Wasserstein analogue on H^n is open research (cite Sturm 2003 *Acta Math* 196:65 for Alexandrov-space framework). Ref: Bauschke-Borwein 2001; Pennec 2020 §2.1.5. (c) ~100 LOC `infogeo/hyperbolic.go` (lives in `infogeo/` not `geometry/`). (d) None.

### H21 — Karcher / Fréchet mean on H^n
(a) Nothing. (b) 206-R15 generic Karcher iteration instantiated on H^n. *No closed-form for n>1* (only two-point case = geodesic midpoint). Generic iteration converges quadratically; injective radius is `+∞` on hyperbolic (no antipodal singularity, unlike sphere). Ref: Karcher 1977; Hauberg-Lauze-Pedersen 2013 *J Math Imaging Vis* 46:103. (c) ~30 LOC instantiation. (d) None beyond 206-R15.

### H22 — Hyperbolic Bregman cone (info-geometry intersection)
(a) Nothing. `infogeo/bregman.go` Euclidean only. (b) Bauschke-Borwein 2001 *SIAM Rev* 43:1 §6: Bregman cone on Lorentz with generator `φ(x) = (⟨x,x⟩_L)^{1/2}` or hyperbolic entropy. Closed-form Bregman projection onto Lorentz cone via spectral decomposition. Extend `infogeo/bregman.go` to accept `Cone` interface; ship `LorentzCone`. Ref: Bauschke-Borwein 2001; Hsieh-Combettes-Pesquet 2018 *J Optim Theory Appl* 178:891. (c) ~80 LOC `infogeo/cone_lorentz.go`. (d) None; sub-link to 092-T1.5.

### H23 — Lorentzian manifold (relativity cross-link)
(a) Nothing. `physics/` ships Newtonian only. (b) Indefinite-metric Lorentzian-manifold structure (signature `(−,+,+,+)`), distinct from Riemannian H^n (positive-definite tangent metric): MinkowskiSpace (cone `⟨x,x⟩_L = 0` = light cone, inside timelike, outside spacelike). Cross-link to 174-physics-relativity if scoped: Lorentz boosts (H9) ARE relativistic boosts; spacetime intervals = Minkowski inner. General Lorentzian = sign-`(−,+,…,+)` generalisation of 207-D2. Ref: MTW 1973 §1-§2; Wald 1984 §3. (c) ~120 LOC `geometry/hyperbolic/lorentzian.go` or `physics/relativity/minkowski.go` per 174-scoping. (d) Placement-only; no implementation blocker.

---

## 2. Implementation summary

| ID | Primitive | LOC | File | Reference |
|----|-----------|-----|------|-----------|
| H1 | Lorentz model H^n | 80 | geometry/hyperbolic/lorentz.go ★ | Ratcliffe 2019 §3 |
| H2 | Lorentz Exp/Log/Distance | 80 | geometry/hyperbolic/lorentz.go ★ | Pennec 2020 §2.1 |
| H3 | Poincaré ball B^n | 80 | geometry/hyperbolic/poincare.go ★ | Cannon-FKP 1997 §2.4 |
| H4 | Möbius arithmetic + matvec | 140 | geometry/hyperbolic/mobius.go | Ungar 2008 §3 |
| H5 | Klein / Beltrami-Klein | 80 | geometry/hyperbolic/klein.go | Cannon-FKP 1997 §2.2 |
| H6 | Upper half-space | 80 | geometry/hyperbolic/halfspace.go | Beardon 1983 §7 |
| H7 | Bijections (4 pairs, 8 fns) | 120 | geometry/hyperbolic/bijection.go ★ | Ratcliffe 2019 §6.3 |
| H8 | Parallel transport closed-form | 70 | geometry/hyperbolic/transport.go | Lorenzi-Pennec 2014 |
| H9 | Möbius transformations / isometries | 120 | geometry/hyperbolic/isometry.go | Ratcliffe 2019 §3.2 |
| H10 | Sarkar tree-embedding | 180 | geometry/hyperbolic/sarkar.go | Sarkar 2011 |
| H11 | Distortion bounds | 230 | geometry/hyperbolic/distortion.go | LLR 1995; Sala 2018 |
| H12 | Nickel-Kiela 2017 graph embedding | 220 | geometry/hyperbolic/embed_graph.go | Nickel-Kiela 2017 |
| H13 | Tifrea 2019 word embedding | 150 | geometry/hyperbolic/embed_word.go | Tifrea 2019 |
| H14 | Hyperbolic Voronoi clustering | 180 | geometry/hyperbolic/voronoi.go | Boguñá 2021 |
| H15 | Hyperbolic NN primitives | 260 | geometry/hyperbolic/nn.go | Ganea-Bécigneul 2018 |
| H16 | Mixed-curvature embedding | 150 | geometry/hyperbolic/mixed.go | Skopek 2020 |
| H17 | Hyperbolic PCA instantiation | 50 | geometry/hyperbolic/pca.go | Fletcher 2004 |
| H18 | Klein chord-path utility | 30 | geometry/hyperbolic/visualise.go | utility |
| H19 | Hyperbolic random sampling | 150 | geometry/hyperbolic/sample.go | Mathieu 2019 |
| H20 | Hyperbolic divergences | 100 | infogeo/hyperbolic.go | Bauschke 2001 |
| H21 | Karcher mean on H^n | 30 | geometry/hyperbolic/karcher.go | Karcher 1977 |
| H22 | Lorentz Bregman cone | 80 | infogeo/cone_lorentz.go | Bauschke-Borwein 2001 |
| H23 | Lorentzian-manifold relativity link | 120 | geometry/hyperbolic/lorentzian.go | MTW 1973 |
|    | **Total Tier-1 (H1+H2+H3+H7)** | **~360** | | |
|    | **Total full canon** | **~3,400** | | |

★ = keystone (H1+H2 Lorentz contract; H3 Poincaré contract; H7 cross-validation via bijection-isometry).

---

## 3. Tier ordering

**Tier 1 (~360 LOC, ship first):** H1 Lorentz model + H2 Exp/Log/Distance + H3 Poincaré ball + H7 Lorentz↔Poincaré bijection. Hyperbolic distance works in both numerical-robust Lorentz and ML-standard Poincaré with cross-model isometry validation; closed-form Exp/Log avoids ODE drift; ready for any downstream consumer.

**Tier 2 (~410 LOC):** H4 Möbius (required by H15 hyperbolic NN) + H5 Klein (required by H14 Voronoi) + H6 Half-space (completes 4-model canon) + H8 parallel transport (required by R-CG/R-L-BFGS) + H7-extended (full bijection canon). Brings hyperbolic surface to Geomstats/Pymanopt parity.

**Tier 3 (~750 LOC, applications):** H10 Sarkar + H11 distortion bounds + H21 Karcher (blocked on 206-R15) + H17 PCA (blocked on 206-R16) + H19 sampling + H22 Bregman cone + H20 divergences. The "low-distortion-tree-embedding" headline lands here.

**Tier 4 (~640 LOC, ML/DL):** H12 Nickel-Kiela graph + H13 Tifrea word + H15 Ganea-Bécigneul-Hofmann NN. The "hyperbolic deep learning" canon. Pairs with 206-R13 R-SGD.

**Tier 5 (~620 LOC, niche):** H9 isometries + H14 Voronoi (sub-blocked on Euclidean Voronoi) + H16 mixed-curvature + H23 Lorentzian-relativity (174-physics cross-link) + H18 visualisation utility.

---

## 4. Architectural recommendations

**A1. New sub-package `geometry/hyperbolic/`** mirroring `optim/proximal/`+`optim/transport/` precedent. ~3,400 LOC full / ~360 LOC Tier 1. One file per model + one file per algorithm. `doc.go` gives model-equivalence theorem; recommends Lorentz default.

**A2. Cross-package consumption.** `optim/manifold/hyperbolic.go` (206-R6) IMPORTS this slot's H1+H2 and decorates with `Manifold` interface (~30-LOC wrapper). `infogeo/hyperbolic.go` (H20) and `infogeo/cone_lorentz.go` (H22) import H1. `geometry/quaternion.go` (existing S³) and `geometry/hyperbolic/lorentz.go` sit side-by-side as the two constant-curvature models.

**A3. Cycle-free DAG.** `geometry/hyperbolic/` → math stdlib only for H1-H8 core; → `graph/` for H10/H12 consumers. `optim/manifold/`, `infogeo/` are downstream consumers. No cycles.

**A4. Default model: Lorentz.** Numerically robust (no boundary singularity), closed-form Exp/Log simpler (single hyperbolic-trig Taylor switch), cross-language parity easier (no `1/(1−‖x‖²)` factors). API: top-level `NewLorentz(n)` default; `NewPoincare(n)` alternative. Both share interface.

**A5. Cross-language parity contract.** Six golden files: (i) `lorentz_exp_log_roundtrip.json` 200 vectors `‖v‖_L ∈ {0, 1e-6, 1e-3, 0.1, 1, 5, 10}`, check `‖Log_x(Exp_x(v)) − v‖_L < 1e-12`; (ii) `poincare_distance.json` 100 (x,y) pairs in B^d for d ∈ {2,5,10}, 1e-13; (iii) `bijection_isometry.json` 100 pairs, transform `L→P`, check `d_P(L→P(x), L→P(y)) = d_L(x,y)` to 1e-12 — *the cross-model test catching sign errors*; (iv) `mobius_associativity.json` 50 (x,y,z) triples, gyrovector axiom `(x⊕y)⊕z = x⊕(y⊕gyr[y,x]·z)` to 1e-13; (v) `sarkar_distortion.json` 30 trees, worst-case `(1+ε)` matches Sala 2018 bound within 5%; (vi) `karcher_consistency.json` 50 sample sets, `‖Σ_i Log_μ(x_i)‖ < 1e-10`.

**A6. Numerical-stability mandate on Taylor fallbacks.** Every `‖v‖→0` switching threshold documented: Lorentz Exp `‖v‖_L < 6e-8` triggers `sinh(t)/t = 1+t²/6+…`; Poincaré-distance Taylor when `u = ‖x−y‖²/((1−‖x‖²)(1−‖y‖²)) < 1e-7` uses `d ≈ √(2u) − u^{3/2}/(3√2) + …`. Cross-language parity brittle if Go uses 1e-4 and C++ uses 1e-8.

**A7. Boundary-handling on Poincaré.** Gradient steps can shoot outside `‖x‖<1`; standard Nickel-Kiela 2017 §3 post-step projection `x ← x · min(1, (1−eps)/‖x‖)` with `eps=1e-5`. Document as part of retraction (not side-effect).

**A8. Zero-alloc hot path.** All Exp/Log/Distance/PT/Möbius consume caller-provided `out []float64` buffers; graph-embedding loops iterate millions of times.

---

## 5. Risks / gotchas

**R1.** Lorentz vs. Poincaré 5–10× wallclock: Lorentz operates in n+1 dim but each step cheaper (no `1/(1−‖x‖²)`); for batch ops Lorentz up to 5× faster.

**R2.** Möbius addition non-commutative AND non-associative; `x⊕y = gyr[x,y]·(y⊕x)`. Test: pin `(x⊕y)⊕z ≠ x⊕(y⊕z)` to catch silent assumptions.

**R3.** **No antipodal singularity on H^n** — unlike S^{n−1} where `Log_x(−x)` direction-ambiguous, on H^n `Log_x(y)` always uniquely defined (injective radius `+∞`). *Strict improvement* over sphere.

**R4.** Boundary singularity in Poincaré at `‖x‖→1` is NOT optional — without `(1−eps)/‖x‖` projection, gradient steps near boundary explode; conformal factor `(1−‖x‖²)^{-1}` blows up. Lorentz has NO boundary singularity (hyperboloid extends to infinity in ambient ℝ^{n+1}).

**R5.** Sarkar requires deg_max-bounded scaling; `τ ≥ 1+log(b)/ε`. For high-deg trees τ large, embedding compresses far from origin, *floating-point precision becomes binding*. Sala 2018 §3 documents `1+O(2^{-d/log(b)})` precision-vs-dimension tradeoff.

**R6.** Klein NOT conformal; angles distorted (only distances/colinearity visually faithful). Klein metric has non-trivial `xxᵀ` direction-dependent component.

**R7.** Distortion-bound tightness varies per tree; complete-binary trees of depth d are tight; most trees achieve much less. Ship `MeasureDistortion` for empirical audit.

**R8.** Hyperbolic-NN composition non-trivial; stacking 10+ layers with `tanh`/`arctanh` activation pipeline requires careful initialisation per Ganea 2018 §5 stable-init scheme.

**R9.** **Cross-language parity at `arccosh(1+ε)` boundary** — high-cancellation expression; different stdlibs have different precision near 1. Tolerance scaling: 1e-12 abs when `ε > 1e-6`, 1e-10 when `ε > 1e-9`, 1e-8 when `ε ≥ 0`.

**R10.** Hyperbolic-Voronoi sub-blocked on Euclidean Voronoi (currently absent in `geometry/`). Either ship Euclidean Voronoi separately (~250 LOC) or use brute-force `O(n²)` for small n.

**R11.** Hyperbolic word-embedding gradient instability at zero — Tifrea 2019 §4: init at exactly origin causes nan because Poincaré-distance gradient has `1/d_B(x,y)` factor exploding when embeddings collide. Standard: small-Gaussian init `N(0, 1e-3·I)`.

**R12.** Mixed-curvature κ-learning instability — Skopek 2020 §4: jointly learning `κ_i` causes `κ→0` collapse (all components flatten to Euclidean). Mitigation: separate κ-update step or constrain `κ_i ∈ [κ_min, κ_max]`.

---

## 6. Cross-package coupling

| Edge | LOC | Purpose |
|------|-----|---------|
| `geometry/hyperbolic/` → math stdlib | 0 | H1-H8 core unblocked |
| `geometry/hyperbolic/` → `graph/` | 0 (call) | H10 Sarkar, H12 graph embed |
| `optim/manifold/hyperbolic.go` (206-R6) → `geometry/hyperbolic/lorentz.go` | 0 | 206 wraps 226 |
| `optim/manifold/karcher.go` (206-R15) → `geometry/hyperbolic/lorentz.go` | 0 | H21 = R15 instantiation |
| `optim/manifold/rpca.go` (206-R16) → `geometry/hyperbolic/lorentz.go` | 0 | H17 = R16 instantiation |
| `infogeo/hyperbolic.go` (H20) → `geometry/hyperbolic/lorentz.go` | 0 | hyperbolic divergences |
| `infogeo/cone_lorentz.go` (H22) → `infogeo/bregman.go` + `geometry/hyperbolic/` | 30 | Bregman cone extension |
| `geometry/hyperbolic/voronoi.go` (H14) → ?-eucl-voronoi | 0 | sub-blocker |
| `physics/relativity/minkowski.go` → `geometry/hyperbolic/lorentzian.go` | 0 | H23 174-cross-link |

Total connective LOC: ~30 (only Bregman cone extension). All other edges call-only — module unusually self-contained.

---

## 7. Single-highest-leverage 1-day project

**Tier-1 H1+H2+H3+H7 (~360 LOC) = `geometry/hyperbolic/{lorentz.go, poincare.go, bijection.go}`.**

1. Closes hyperbolic-distance gap with one PR — `LorentzDistance(x,y)` / `PoincareDistance(x,y)` with cross-model isometry validation.
2. Validates negative-curvature contract on two industry-standard models. Both 1e-12 round-trip and 1e-12 bijection-isometry → rest of canon extends cleanly.
3. Pure additive surface; no break to `geometry/`/`optim/`/`infogeo/`.
4. Unblocks 206-R6 (30-LOC Manifold wrapper) and 092-T2.5 (30-LOC instantiation).
5. Cross-language parity test: 3 golden grids (~400 vectors, 1e-12 tolerance), reproducible from first-principles in Go/Python/C++/C#.

---

## 8. Single-highest-leverage cutting-edge piece

**Tier-3 H10 Sarkar (180 LOC) + H11 distortion bounds (230 LOC) = ~410 LOC for tree → low-distortion hyperbolic embedding.**

1. **Killer feature of hyperbolic vs Euclidean.** Sarkar 2011 + Sala 2018: any tree embeds into Poincaré disk with `(1+ε)` distortion for arbitrarily small ε. Use cases: hierarchical-data compression (2D Poincaré encodes any tree with bounded distortion vs Ω(log n) Euclidean lower bound), citation-graph layout, taxonomic/phylogenetic tree analysis, JSON-tree distance metrics.
2. **No mainstream pure-math library ships Sarkar with closed-form bounds.** Geomstats has Poincaré-ball but not Sarkar; PyTorch-Geometric has gradient-learning hyperbolic embeddings (NOT closed-form Sarkar). Reality would be the *only* zero-dependency pure-math library with Sarkar + analytical bounds.
3. Pairs with H11 — closed-form theoretical bounds (LLR 1995, Bartal 1996, Sarkar 2011, Sala 2018) shipped as runtime functions. Theoretical-vs-empirical gap auditable.
4. Strong test contract. Sarkar(complete-binary-tree-depth-10, ε=0.01) → embedding `f`; check `d_B(f(u), f(v))/d_T(u,v) ∈ [1, 1+ε]` all pairs; check `‖f(v)‖ < 1−eps` no boundary violation; check distortion vs Sala 2018 bound. ~50 LOC golden tests.
5. Cross-link to `graph/` light — input is `graph.Graph` with `IsTree()` precondition; output `[]Vec` per node. No deep coupling.

---

## 9. Verdict

**SHIP** Tier 1 (~360 LOC) — Lorentz + Poincaré + bijection. Validates negative-curvature canon on two industry-standard models with cross-model isometry pin. Unblocks 206-R6, 092-T2.5, 207-D9 hyperbolic-plane closed-form validation.

**SHIP** Tier 2 (~410 LOC) — Möbius + Klein + Half-space + parallel transport + full bijection canon. Brings to Geomstats/Pymanopt parity; pairs naturally with Tier 1, can ship same sprint if bandwidth permits.

**SHIP** Tier 3 (~750 LOC) — Sarkar + distortion bounds + Karcher + PCA + sampling + Bregman cone + divergences. The "low-distortion-tree-embedding" headline lands here. Soft cross-link to 206-R15/R16.

**SHIP** Tier 4 (~640 LOC) — Nickel-Kiela + Tifrea + Ganea-Bécigneul-Hofmann NN. Hyperbolic deep-learning canon; pairs with 206-R13.

**SHIP-WHEN-CONSUMER-PULLS** Tier 5 (~620 LOC) — isometries + Voronoi (sub-blocked on Euclidean Voronoi) + mixed-curvature + Lorentzian-relativity + visualisation.

**Cross-slot synergy:**
- **092-T2.5**: this slot IS the implementation 092 punted to v2; T2.5 reduces to instantiation+092-T2.1 Manifold-decoration+092-T1.2 Fisher-Rao cross-link.
- **206-R6**: 250-LOC `Manifold` instance reduces to 30-LOC wrapper around this slot's H1+H2+H8.
- **207-D9**: generic geodesic-ODE invokes 226-H2 closed-form when metric detected as constant-negative-curvature (5–8× speedup, exact).
- **177-synergy / 107-graph-missing**: did not scope hyperbolic; this slot purely additive.
- **174-physics-T-relativity** (if scoped): 226-H23 shares `LorentzBoost` primitive — single-source-of-truth here in `geometry/hyperbolic/`.
- **092-T1.5** (generic Bregman): 226-H22 is *first non-Euclidean-flat* Bregman generator; recommend extending `infogeo/bregman.go` to accept `Cone` interface.
- **204-symplectic**: hyperbolic-geodesic Hamiltonian preserves `‖p‖_L²`; closed-form Exp on H^n makes symplectic mostly unnecessary, but cross-validation testing benefits.
- **011/012/013-autodiff**: forward-mode autodiff required for *generic-numerical* path of hyperbolic-NN backward-pass (H15); closed-form per-layer (Möbius matvec backward) ships unblocked.

The hyperbolic-embedding canon adds ~3,400 LOC to Reality as a self-contained `geometry/hyperbolic/` sub-package with **zero hard cross-package blockers** for Tiers 1+2 (~770 LOC = first-sprint shippable). The "tree → low-distortion Poincaré-disk" headline (H10+H11, ~410 LOC, Tier 3) is the cutting-edge cross-language parity demonstration that no mainstream library ships cleanly. Cross-language parity contract shippable on H1+H2+H3+H7 core (~360 LOC) with 6 golden-file grids covering round-trip Exp/Log, bijection isometry, distance closed-forms, Möbius axioms, Sarkar distortion, Karcher consistency.

---

*226-new-hyperbolic-embed.md — 308 lines.*

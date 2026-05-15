# 276 — new-cvxopt-extras (Block C: convex-optimisation extras)

**Two-line summary.** reality v0.10.0 ships LP-only convex programming (`optim.SimplexMethod` + `optim.InteriorPoint`-as-barrier-gradient) plus first-order proximal/ADMM scaffolding (`optim/proximal/{Fbs,Fista,Admm,ProxL1/L0/SquaredL2/NonNeg/Box/L2Ball/Simplex/Linear}`); the entire QP/SOCP/SDP/conic/QCQP/Frank-Wolfe/cutting-plane/bundle/smoothing/DC/distributionally-robust + the convex-relaxation consumers (Goemans-Williamson MaxCut, Lovász θ-via-SDP, matrix-completion-via-nuclear-norm-SDP) are absent. Slot 276 enumerates 24 primitives X1–X24 (~3,180 LOC) organised in 7 tiers; cross-links to 102-T1.13/T1.14/T2.10/T3.1/T3.2 (the optim-missing tier-1+tier-3 conic asks), 174-G18 (Frank-Wolfe in OCO), 199-G6 (Lovász θ blocked on SDP), 259-M1-M6 (matrix-completion-via-SDP-nuclear-norm), 169-synergy-prob-optim (DRO ambiguity sets); recommended placement `optim/conic/` + `optim/cvx/` sub-packages.

---

## 1. Inventory: what reality has today

### 1.1 LP layer (`optim/linear.go`, 316 LOC)
- **`SimplexMethod(c, A, b)`** — revised simplex with Bland's anti-cycling rule, `maxIter=10000`. Standard form `min c'x s.t. Ax≤b, x≥0`. **Status: production-grade for small dense LP, no warm-start, no presolve.**
- **`InteriorPoint(c, A, b)`** — same standard form. Mislabelled. Read lines 266-273: the "Newton step" is `dx[j] = -rd[j] * x[j]^2 / (mu + x[j]^2)` — this is **not Newton on KKT**, it is a damped gradient step on the log-barrier, no Cholesky-backsolve of the augmented system, no Mehrotra predictor-corrector. Slot 101-F7 + 102-T1.13 already filed to **rename → `BarrierGradientHeuristic` and replace with Mehrotra MPC** (~250 LOC, reuses `linalg.CholeskyDecompose`).

### 1.2 Proximal / first-order layer (`optim/proximal/`, ~600 LOC)
- **`Fbs(grad, prox, x, work, cfg)`** — forward-backward splitting / proximal gradient.
- **`fistaLoop`** — Beck-Teboulle 2009 FISTA acceleration.
- **`Admm(proxF, proxG, x, z, u, cfg)`** — consensus ADMM `min f(x)+g(z) s.t. x=z` (Boyd 2011 §3.1.1). Scaled-form, infinity-norm primal+dual residual termination. **No general-form ADMM `min f(x)+g(Ax)`; no over-relaxation; no adaptive ρ.**
- **Prox library:** `ProxL1` (soft-thresh), `ProxL0` (hard-thresh, non-convex), `ProxSquaredL2`, `ProxNonNeg`, `ProxBox(lo,hi)`, `ProxL2Ball(r)`, `ProxSimplex` (sort-based Held-Hochbaum-Wolfe 1974), `ProxLinear(c)`. **Missing:** `ProxNuclear` (singular-value soft-thresh, gates 259-M1 SVT), `ProxSemidefinite` (PSD projection, gates SDP), `ProxSecondOrderCone` (cone projection, gates SOCP), `ProxBoxL1` etc.

### 1.3 What is **absent** (verified `grep -i "QP|SOCP|SDP|conic|Mehrotra|MaxCut|Lovasz|FrankWolfe|QCQP|CuttingPlane|Bundle|Smoothing|Distributionally|RobustOpt"` over reality `*.go` returns 0 callable matches in production code outside doc-only mentions):

| Family | Status |
|---|---|
| Quadratic Program (QP) `min ½xᵀQx+cᵀx s.t. Ax≤b, Cx=d` | **ABSENT.** Listed as 102-T1.14 (~300 LOC OSQP-style ADMM). |
| Active-set QP (Goldfarb-Idnani 1983) | **ABSENT.** 102-T2.35. |
| Primal-dual interior-point QP (Wright 1997) | **ABSENT.** 102-T2.34. |
| Second-Order Cone Program (SOCP) | **ABSENT.** 102-T3.1 (~600 LOC Nesterov-Todd IPM). |
| Semidefinite Program (SDP) | **ABSENT.** 102-T3.2 (~800 LOC AHO direction IPM). |
| Conic standard form `min cᵀx s.t. Ax=b, x∈K` | **ABSENT** — no abstraction over LP/SOCP/SDP/exp-cone hierarchy. |
| Self-dual embedding (Ye-Todd-Mizuno 1994) | **ABSENT.** No homogeneous self-dual model anywhere. |
| Mehrotra predictor-corrector (1992) | **ABSENT.** 102-T1.13. |
| SCS / Splitting Conic Solver (O'Donoghue 2016) | **ABSENT.** Conic ADMM. |
| ECOS interior-point conic (Domahidi 2013) | **ABSENT.** |
| Quadratically Constrained QP (QCQP) | **ABSENT.** |
| DC programming / DCA (Tao 1986) | **ABSENT.** |
| Proximal point algorithm (PPA, Rockafellar 1976) | **ABSENT** (FBS/FISTA are *prox-gradient*, distinct). |
| Frank-Wolfe / conditional gradient | **ABSENT.** Listed in 174-G18 OCO online-form, not as offline algorithm. |
| Sub-gradient method | **ABSENT.** |
| Cutting-plane / Kelley 1960 | **ABSENT.** |
| Bundle method (Lemaréchal 1975, Kiwiel 1990) | **ABSENT.** |
| Smoothing techniques (Nesterov 2005) | **ABSENT.** |
| Distributionally Robust Opt (Wasserstein-DRO, Mohajerin-Esfahani-Kuhn 2018) | **ABSENT.** |
| Robust optimisation (Ben-Tal-Nemirovski 1998, Bertsimas-Sim 2004) | **ABSENT.** |
| Goemans-Williamson 1995 SDP for MAX-CUT | **ABSENT.** Gates 254-graph-cuts approx. |
| Lovász θ via SDP | **ABSENT.** Gates 199-G6 (only eigenvalue bound shippable today). |
| Matrix-completion via nuclear-norm SDP | **PARTIAL** — 259-M1-M6 specifies SVT/ADMM but proxNuclear missing. |
| Trace + eigenvalue constraints | **ABSENT.** |
| Convex relaxation library | **ABSENT.** |

### 1.4 Pre-existing cross-link surface
- **097-T1** asks for `linalg.Eigvec` (eigenvectors via inverse iteration on QR-shifted form) — required for SDP PSD-projection in inner loop.
- **102-T1.13** Mehrotra MPC LP — direct prerequisite for any IPM SOCP/SDP.
- **102-T1.14** OSQP-style QP ADMM — reuses `optim/proximal/admm.go` infrastructure.
- **102-T3.1** SOCP IPM — Nesterov-Todd direction.
- **102-T3.2** SDP IPM — AHO direction. Requires symmetric eigendecomposition (already in `linalg.QRAlgorithm`).
- **174-G18** Frank-Wolfe (OCO online form). Slot 276 owns offline batch FW.
- **199-G6** Lovász θ — eigenvalue sandwich (~80 LOC) ships now, full SDP blocked behind 276.
- **254-graph-cuts** ships max-flow/min-cut for image-segmentation (network flow tier); GW-MaxCut SDP relaxation is the **approximation-algorithm** companion that 254 explicitly does NOT cover.
- **259-M1-M6** matrix-completion-via-SVT/ADMM — needs `ProxNuclear` from this slot.
- **169-synergy-prob-optim** flagged distributionally-robust optimisation as joint prob×optim seam.

---

## 2. The 24-primitive proposal X1–X24 (~3,180 LOC)

### Tier 0 — Conic standard-form abstraction (~280 LOC, foundational)
**X1 — `optim/conic/Cone` interface (~80 LOC).** The single keystone abstraction. Each cone implements `Project(v, out)` (Euclidean projection), `IsInterior(v) bool`, `LogBarrier(v) float64`, `Dim() int`. Concrete cones:
- `Zero{n}` (equality)
- `NonNeg{n}` (LP)
- `SecondOrderCone{n}` — `{(t,x) : ‖x‖₂ ≤ t}`, projection via radial scaling (Boyd-Vandenberghe 2004 §8.1.1)
- `PSDCone{n}` — `{X : X⪰0, n×n symmetric}`, projection via spectral truncation (eigendecompose, clip negatives, reconstruct) reuses `linalg.QRAlgorithm` + needs eigenvectors (097-T1 hard-blocker)
- `ExpCone` — `{(x,y,z) : y·exp(x/y) ≤ z, y>0}`, projection via Newton on 1-D characteristic equation (Friberg 2021)
- `PowerCone{α}` — `{(x,y,z) : x^α·y^(1-α) ≥ |z|, x,y≥0}`

**X2 — `ConicProgram` standard form (~80 LOC).** `min cᵀx s.t. Ax=b, x∈K = K₁ × K₂ × … × Kₘ` (product cone). Validates conformability, packs/unpacks block-structured `x`. Self-dual.

**X3 — Cone duality utilities (~60 LOC).** `DualCone(K)`, `IsSelfDual(K)`, `BlockProject(x, cones, out)`. Foundational for primal-dual algorithms downstream.

**X4 — Conic LP equivalence (~60 LOC).** `LPToConic(c, A, b)` lifts simplex-form LP into cone-form (just `K = NonNeg`); enables single solver path.

### Tier 1 — Quadratic programming (~520 LOC, fills 102-T1.14)
**X5 — `OSQP-style QP ADMM` (~280 LOC).** Stellato-Banjac-Goulart-Bemporad-Boyd 2020 *Math. Prog. Comput.* 12:637. `min ½xᵀPx + qᵀx s.t. l ≤ Ax ≤ u`. Outer ADMM splits `x` from auxiliary `z = Ax`. Inner step is one KKT linear solve per iteration with a **factorisation-cached** symmetric quasi-definite system — reuses `linalg.CholeskyDecompose` on `[P+σI, Aᵀ; A, -1/ρ I]` after LDLT promotion. Adaptive `ρ` rule (every 25 iter rebalance primal/dual). **Saturated by R-OSQP-VS-INTERIOR-POINT-3/3 pin (small QP cross-validate against X6).**

**X6 — Goldfarb-Idnani active-set QP (~160 LOC).** *Math. Prog.* 27:1, dual method for strictly convex QP with `Ax ≥ b`. Adds/drops constraints one at a time; cheap small-problem alternative. Polynomial in m, exponential worst-case in n but practically fast for n≤50.

**X7 — Primal-dual IPM QP (~80 LOC, alternative path).** Wright 1997 §16.6, Mehrotra predictor-corrector applied to QP KKT. Reuses X12/X13 once Mehrotra LP lands.

### Tier 2 — Mehrotra LP + IPM-conic (~640 LOC, fills 102-T1.13/T3.1)
**X8 — `MehrotraPredictorCorrector` LP (~280 LOC).** *SIAM J. Optim.* 2:575, 1992. Industrial-grade primal-dual IPM. Replaces (or sits beside) `optim.InteriorPoint`. Three Newton solves per iteration — affine, centring, corrector — each a normal-equation Cholesky `(AΘAᵀ)Δλ = r`. Mehrotra heuristic for `μ` and step length. **Single most-cited LP algorithm of the past 30 years** (used as default of MOSEK/CPLEX). 102-T1.13 already specifies this.

**X9 — Self-dual embedding (Ye-Todd-Mizuno 1994 *Math. Oper. Res.* 19:53) (~120 LOC).** Homogeneous self-dual model `min 0 s.t. M·z + s = q, z,s ≥ 0` where `M = [0,A,-b; -Aᵀ,0,c; bᵀ,-cᵀ,0]`. Always strictly feasible interior point (start from `e`), automatic infeasibility detection (no Phase-I needed). Critical infrastructure for X11 SCS.

**X10 — SOCP via Nesterov-Todd scaling (~240 LOC).** Nesterov-Todd 1997 *Math. Oper. Res.* 22:1, primal-dual self-scaled IPM specialised to second-order cone. Scaling matrix `W = (1/(s∘x))·I + …` block-diagonal across cones. Reuses X8 Mehrotra structure but with cone-specific projections from X1. 102-T3.1 ask. Foundational for portfolio optimisation, robust LS, antenna design (cross-link 215-compressed-sensing).

### Tier 3 — Semidefinite programming + SCS (~620 LOC, fills 102-T3.2)
**X11 — `SCS` Splitting Conic Solver (~340 LOC).** O'Donoghue-Chu-Parikh-Boyd 2016 *J. Optim. Theory Appl.* 169:1042. **First-order conic ADMM on the homogeneous self-dual embedding (X9).** Each iteration: (1) projection onto subspace `{(u,v) : Q·u = v}` via cached LDL factorisation of `[I, Aᵀ; A, -I]`, (2) projection onto cone product `K × K* × R₊`. Ships SDP without symmetric eigendecomposition in inner loop (PSD projection reuses linalg every iter — still needed but as oracle, not as factor). Modest precision (1e-3 to 1e-5) — pairs with X12 SDP-IPM for high-precision warm-start. **Single most-deployed open-source conic solver in the world (cvxpy default, used in MOSEK ground-truth comparisons).**

**X12 — SDP via AHO direction (~280 LOC).** Alizadeh-Haeberly-Overton 1998 *SIAM J. Optim.* 8:746. Primal-dual IPM for `min ⟨C,X⟩ s.t. ⟨Aᵢ,X⟩=bᵢ, X⪰0`. Symmetric AHO Newton system `EΔX·F + EᵀΔX·Fᵀ = R` solved via Lyapunov-equation reformulation. Each iteration calls `linalg.QRAlgorithm` for symmetric eigendecomposition of `n×n` matrix — currently O(n³) per iter. **Hard-blocker on 097-T1 (eigenvectors).** 102-T3.2 ask. Eight to sixteen iterations typical for PSD-feasibility plus one optimisation phase.

### Tier 4 — Convex-relaxation consumers — THE singular moat (~440 LOC)
**X13 — Goemans-Williamson MaxCut (~120 LOC).** Goemans-Williamson 1995 *J. ACM* 42:1115, **0.87856-approximation** for MAX-CUT (and the longstanding tightest unconditional approximation ratio for any APX-hard problem). Solve SDP `max ½ Σ_{(i,j)∈E} (1 − Xᵢⱼ) s.t. Xᵢᵢ=1, X⪰0` (uses X12 or X11), Cholesky-factor `X = VᵀV`, sample random hyperplane `r` and round `xᵢ = sign(rᵀvᵢ)`. **Singular pedagogical keystone of approximation-algorithm theory.** Cross-link 254-graph-cuts as the relaxation companion to algorithmic max-flow/min-cut.

**X14 — Lovász θ via SDP (~80 LOC).** `θ(G) = max{⟨J,X⟩ : Xᵢⱼ=0 ∀{i,j}∈E, tr(X)=1, X⪰0}`. Lovász 1979 *IEEE Trans. Inform. Theory* 25:1. Tight sandwich `α(G) ≤ θ(G) ≤ χ̄(G)` for perfect graphs. **Unblocks 199-G6 fully** — currently 199 ships only the eigenvalue bound (~80 LOC stub) explicitly noting blocked-on-SDP.

**X15 — Matrix completion via nuclear-norm SDP (~120 LOC).** Candès-Recht 2009 *Found. Comput. Math.* 9:717. Standard formulation `min ‖X‖_* s.t. P_Ω(X) = P_Ω(M)` cast as SDP via `‖X‖_* = ½ min{tr(W₁)+tr(W₂) : [W₁,X; Xᵀ,W₂]⪰0}`. **Lifts 259-M1 SVT to globally-optimal high-precision completion.**

**X16 — Quadratically Constrained QP (QCQP) via SDP relaxation (~120 LOC).** `min xᵀP₀x s.t. xᵀPᵢx ≤ qᵢ` → `min ⟨P₀,X⟩ s.t. ⟨Pᵢ,X⟩ ≤ qᵢ, X⪰0` (dropping rank-1). Boyd-Vandenberghe 2004 §5.2. Plus Shor relaxation tightness witness. Cross-link to 187-orbital trajectory optimisation.

### Tier 5 — Frank-Wolfe + sub-gradient + cutting-plane (~440 LOC)
**X17 — Frank-Wolfe / conditional gradient (~120 LOC).** Frank-Wolfe 1956 *Naval Res. Log. Q.* 3:95, Jaggi 2013 *ICML*. `min f(x) s.t. x ∈ D` (D compact convex). Each iter: `s = argmin_{s∈D} ⟨∇f(x),s⟩` (LP oracle), `x ← (1−γ)x + γs`. Sublinear `O(1/T)` for smooth-convex. **Projection-free** — only LP solves. Specialise to `D = simplex` (X1 oracle) and `D = nuclear-norm-ball` (Hazan 2008). 174-G18 has the OCO variant; 276 owns the offline batch + away-step variant (Lacoste-Julien-Jaggi 2015 linear convergence).

**X18 — Sub-gradient method (~60 LOC).** Polyak 1969. `x_{k+1} = Π_D(x_k − α_k g_k)`, `g_k ∈ ∂f(x_k)`. Diminishing `α_k = 1/√k` step rule. The only first-order method for **non-smooth non-decomposable** convex `f`. Reuses `optim/proximal` projections from operators.go.

**X19 — Cutting-plane / Kelley method (~100 LOC).** Kelley 1960 *J. SIAM* 8:703. `min f(x) s.t. x∈P` for convex `f`: at each iter add subgradient cut `f(x_k) + ⟨g_k, x − x_k⟩ ≤ z`, solve LP master. Pairs with X1 simplex. Foundational for two-stage stochastic LP and Benders decomposition.

**X20 — Bundle method (~160 LOC).** Lemaréchal 1975, Kiwiel 1990 *Math. Prog.* 46:105. Bundle of past sub-gradients + proximal-stabilised cutting-plane master `min ½‖x − x̂‖² + max_i [f(xᵢ) + ⟨gᵢ, x − xᵢ⟩]`. **Practically dominant non-smooth optimiser.** Active-set master QP solved by X6 Goldfarb-Idnani.

### Tier 6 — Smoothing + DC + DRO + robust (~460 LOC)
**X21 — Nesterov smoothing (~100 LOC).** Nesterov 2005 *Math. Prog.* 103:127. Replace non-smooth `f(x) = max_y ⟨Ax,y⟩ - φ(y)` with smooth `f_μ(x) = max_y [⟨Ax,y⟩ - φ(y) - μ d(y)]` (d strongly convex prox-function). Resulting `f_μ` is `1/μ`-Lipschitz-gradient. Apply FISTA → `O(1/√ε)` complexity, near-optimal. **Bridges proximal world to non-smooth saddle-point world.**

**X22 — DC programming / DCA (~120 LOC).** Tao-An 1986, Tao-Pham-Dinh 1997 *Acta Math. Vietnamica* 22:289. `min g(x) − h(x)` (g, h convex). DCA iteration: `x_{k+1} ∈ ∂g*(y_k)`, `y_{k+1} ∈ ∂h(x_{k+1})` (legendre transforms). Convergent to critical point. Specialises to majorisation-minimisation and surrogate concave penalties. Cross-link 232-robust-stats (M-estimator non-convex losses).

**X23 — Distributionally robust optimisation, Wasserstein-ball (~140 LOC).** Mohajerin-Esfahani-Kuhn 2018 *Math. Prog.* 171:115. `min_θ sup_{Q : W₁(Q,P̂) ≤ ε} E_Q[ℓ(θ;ξ)]`. Strong duality reformulation as finite-dimensional convex program for Lipschitz-loss `ℓ`. Foundational for **distributionally-robust ML and robust portfolio**. Pairs with X10 SOCP for Lipschitz constraints. Cross-link 169-synergy-prob-optim.

**X24 — Robust optimisation, Bertsimas-Sim Γ-uncertainty (~100 LOC).** Bertsimas-Sim 2004 *Oper. Res.* 52:35. `min cᵀx s.t. (a + ξ)ᵀx ≤ b ∀ ξ ∈ U`, polyhedral uncertainty `U = {ξ : ‖ξ‖_∞ ≤ 1, ‖ξ‖_1 ≤ Γ}`. Tractable reformulation as LP with `Γ` extra variables. Conservativeness tunable by `Γ ∈ [0,n]`. Pair with Ben-Tal-Nemirovski 1998 ellipsoidal-uncertainty SOCP reformulation. **Industrial workhorse — used in supply-chain, network-design, energy.**

---

## 3. R-MUTUAL-CROSS-VALIDATION saturation pins (5 pins, all 3/3)

| Pin | Primitives | Witness |
|---|---|---|
| **R-LP-CONIC-EQUIVALENCE** | X1 NonNeg + X2 ConicProgram + X8 Mehrotra + existing SimplexMethod | LP solved as `min cᵀx s.t. Ax=b, x∈NonNeg` via X8 must agree byte-equal with SimplexMethod for integer-feasible problems and within 1e-7 for transcendental — sanity-checks conic abstraction. |
| **R-QP-OSQP-VS-IPM-VS-ACTIVESET** | X5 OSQP-ADMM + X6 Goldfarb-Idnani + X7 IPM-QP | Three solvers cross-validate on portfolio mean-variance QP (Markowitz `min ½xᵀΣx s.t. μᵀx≥r, 1ᵀx=1, x≥0`). **The most demanding single witness — saturates QP layer.** |
| **R-SDP-MAXCUT-GW-RATIO** | X11 SCS + X12 AHO-IPM + X13 GW + linalg.CholeskyDecompose | On Frieze-Jerrum random-cubic-graph G(n,3), GW rounding using SDP-from-X11 vs SDP-from-X12 must yield expected cut value within 0.87856·OPT_LP-relaxation ± 1/√n. **Pedagogical keystone witness for SDP correctness.** |
| **R-NUCLEAR-PROX-VS-SDP** | X15 nuclear-norm-SDP + 259-M1 SVT + 259-M2 nuclear-prox | Recover known low-rank `M = uvᵀ + N(0,σ²)` via three orthogonal routes (proximal SVT iteration, ADMM-completion, full nuclear-SDP). All three converge to same Frobenius-error scaling. **Cross-validates 259 against 276.** |
| **R-FW-VS-PROJGRAD-SIMPLEX** | X17 Frank-Wolfe + 174-G18 OCO-FW + ProxSimplex projected-gradient | Constrained smooth-convex on simplex (e.g., min KL-divergence to fixed point in simplex) — three solvers must agree. |

---

## 4. SINGULAR axis classifications

- **SINGULAR-CHEAPEST-1-DAY: X1 + X2 + X3 + X4 + X14 (~360 LOC).** Cone abstraction + LP-as-conic-LP + Lovász-θ-eigenvalue-bound. Zero cross-package blockers (relies only on existing `linalg.QRAlgorithm`). Ships immediate value: every later tier reuses X1-X3, and X14 unblocks 199-G6 immediately at the eigenvalue-sandwich precision.
- **SINGULAR-FOUNDATIONAL: X8 Mehrotra MPC LP + X11 SCS (~620 LOC).** Mehrotra is the single most-cited LP IPM algorithm of the past 30 years (industrial default of MOSEK/CPLEX). SCS is the single most-deployed open-source conic solver in the world (cvxpy default). Without these two, `reality` cannot reach scipy-equivalence on the convex-programming axis.
- **SINGULAR-MOAT: X11 SCS + X12 AHO-SDP + X13 GW-MaxCut + X15 NuclearSDP (~860 LOC).** No zero-dependency MIT-licensed Go conic solver exists worldwide. cvxpy is Python-only, MOSEK/CPLEX/Gurobi are commercial, ECOS is GPL-3, SCS is Apache-2 (compatible). Reality post-PR-Tier-3 would be the **only zero-dep MIT-licensed Go conic optimiser ever shipped**.
- **SINGULAR-2024-FRONTIER: X23 Wasserstein-DRO + X22 DCA + X20 Bundle (~420 LOC).** DRO is the single hottest area in robust ML 2018-2024 (Wasserstein-DRO by Mohajerin-Esfahani-Kuhn 2018 has 2,000+ citations as of 2026). DCA powers nonconvex sparse-regression (SCAD/MCP penalties), bundle methods power L1 SVMs at scale.
- **SINGULAR-PEDAGOGICAL: X8 + X12 + X13 + X17 (~820 LOC).** Mehrotra MPC + AHO SDP + Goemans-Williamson MaxCut + Frank-Wolfe = the four canonical convex-optimisation pillars of every graduate text (Boyd-Vandenberghe, Nesterov-Wright, Williamson-Shmoys, Bubeck).

---

## 5. Cross-package blockers

| Type | Dependency | Impact |
|---|---|---|
| **HARD** | `linalg.Eigvec` (097-T1, currently absent) | Required for X12 AHO-SDP PSD-projection in inner loop. SCS-X11 ALSO needs it via the cone-projection step on PSD blocks. **Single largest gating dependency.** Without 097-T1, all SDP primitives X12+X13+X14+X15 are blocked. |
| **HARD** | `linalg.LDLT` (currently absent — only Cholesky for SPD) | X5 OSQP requires LDLT for symmetric quasi-definite system `[P+σI, Aᵀ; A, -1/ρ I]`. Workaround: ridge-regularise to SPD then Cholesky (loses precision). |
| **MEDIUM** | `linalg.SymmetricEigen` (X12 inner loop) | `linalg.QRAlgorithm` ships symmetric tridiagonal QL — can be wrapped, but lacks eigenvectors. Same bottleneck as 097-T1. |
| **SOFT** | `optim.MehrotraPredictorCorrector` (102-T1.13) | X10 SOCP and X12 SDP both reuse the Mehrotra outer-loop machinery. If 102-T1.13 ships first, X10/X12 become ~30% smaller. |
| **NONE** | `optim/proximal/` ADMM, prox-operators, FISTA | All present. Direct reuse for X5 OSQP-QP and X11 SCS. |
| **NONE** | `linalg.CholeskyDecompose` | Present. Used by X8 Mehrotra normal-eqn solve, X5 OSQP factor-cache. |
| **NONE** | `combinatorics`, `graph` | Present. Used by X13 GW-MaxCut (graph), X23 DRO (sample-mean baseline). |

---

## 6. Recommended placement and PR sequence

**Sub-package layout:**
```
optim/
  conic/
    cone.go         # X1 Cone interface + concrete cones
    program.go      # X2 ConicProgram, X3 duality, X4 LP-to-conic
    socp.go         # X10 NT-SOCP IPM
    sdp.go          # X12 AHO-SDP IPM
    scs.go          # X11 SCS conic ADMM (depends on X9)
    selfdual.go     # X9 self-dual embedding
    cone_test.go    # 30+ vectors per cone
  qp/
    osqp.go         # X5 OSQP-ADMM
    activeset.go    # X6 Goldfarb-Idnani
    ipm.go          # X7 PD-IPM-QP
  cvx/              # convex-optimisation general algorithms
    mehrotra.go     # X8 Mehrotra LP MPC (replaces InteriorPoint)
    framework.go    # X17 Frank-Wolfe, X18 Sub-gradient
    cutting.go      # X19 Kelley, X20 Bundle
    smoothing.go    # X21 Nesterov smoothing
    dca.go          # X22 DC programming
    dro.go          # X23 Wasserstein-DRO
    robust.go       # X24 Bertsimas-Sim
  relax/            # convex-relaxation library (consumers)
    maxcut.go       # X13 GW
    lovasz.go       # X14 Lovász θ
    matcomp.go      # X15 nuclear-norm matrix-completion
    qcqp.go         # X16 QCQP-Shor
```

**6-PR sequence:**
1. **PR-A (1 day, ~360 LOC, ZERO blockers):** X1 + X2 + X3 + X4 + X14-eigenvalue-bound. Conic abstraction + LP-as-conic-LP. Unblocks 199-G6 immediately. Ships `optim/conic/cone.go`, `program.go`.
2. **PR-B (1 week, ~520 LOC):** X5 OSQP-QP + X6 Goldfarb-Idnani. Closes 102-T1.14. Pairs with cross-validation against X7 (deferred).
3. **PR-C (1 week, ~280 LOC):** X8 Mehrotra MPC LP, replacing `optim.InteriorPoint`. Closes 102-T1.13. Renames current `InteriorPoint` → `BarrierGradientHeuristic`.
4. **PR-D (1 week, ~340 LOC):** X11 SCS conic ADMM + X9 self-dual embedding. Lower-precision SDP/SOCP path that does **not** require eigenvectors at high precision (only for projection oracle). Critical: enables X13 + X14-full + X15 at ε=1e-4 precision **even while 097-T1 is still outstanding**.
5. **PR-E (1-2 weeks, ~520 LOC, BLOCKED on 097-T1):** X10 SOCP-NT + X12 AHO-SDP. High-precision IPM path. Ships once linalg has eigenvectors.
6. **PR-F (1 week, ~720 LOC):** X13 GW + X15 nuclear-SDP + X16 QCQP + X17 Frank-Wolfe + X18-X20 cutting-plane/bundle/sub-gradient + X21 smoothing + X22 DCA + X23 DRO + X24 robust. The "consumer + extras" wave.

**Total ship recommended for v0.11–v0.12:** ~3,180 LOC across 24 primitives. PR-A through PR-D (~1,500 LOC) ship **without** 097-T1 unblock; PR-E (~520 LOC) requires 097-T1 first; PR-F (~720 LOC) ships at modest precision via SCS-X11 even without 097-T1.

---

## 7. Candid assessment vs other Block C slots

276 sits in a privileged position: **every Block-C slot that asks for an "SDP solver" or "conic solver" actually means slot 276.** Concretely, 199-G6 (Lovász θ), 254-graph-cuts (GW-MaxCut companion at relaxation tier), 259-M1-M6 (matrix-completion-via-SDP), 215-compressed-sensing (SOCP for L1-LS and Basis Pursuit Denoising), 232-robust-stats (DCA for SCAD/MCP), 169-synergy-prob-optim (Wasserstein-DRO), 187-orbital-control (QCQP for trajectory) all enumerate primitives that **delegate to 276's solvers**. 276 is therefore **the second-highest leverage slot in Block C** behind 097-T1 (`linalg.Eigvec`) which gates 276's tier-3 in turn.

The slot is **not** redundant with: (1) `optim/proximal` (276 uses it as substrate, but proximal is *first-order generic*, not conic), (2) `optim.SimplexMethod` (276 lifts simplex into conic-LP form via X4), (3) 102-optim-missing (102 enumerates the *primitive list*; 276 enumerates the *unified conic + relaxation + DC + DRO architecture* with cross-validation pins and concrete sub-package layout — a strict superset of 102-T1.13/T1.14/T2.10/T2.34/T2.35/T3.1/T3.2 plus the relaxation-consumer + DC/DRO/robust/smoothing/cutting-plane axes that 102 skipped or stubbed).

**Verdict.** PR-A (1 day) + PR-D (1 week) ship the bulk of unique value (conic abstraction + SCS) without requiring the 097-T1 unblock and **immediately deliver** SDP-based MaxCut, Lovász θ, matrix-completion at modest precision. PR-B + PR-C close the QP and LP gaps respectively. PR-E + PR-F are the "research-grade" wave gated on 097-T1. **Tier-1 mathematical-canonical-importance, Tier-1 implementation-priority** — recommend slot 276 ship at v0.11.0 alongside or just after 097-T1.

---

*Report length: ~280 lines including prose tables.*

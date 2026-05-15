# 242 | new-spde — Stochastic Partial Differential Equations

**Summary line 1.** reality v0.10.0 ships **zero** SPDE machinery — repo-wide grep on `KPZ|Kardar.?Parisi.?Zhang|stochastic.heat|stochastic.Burgers|cylindrical.Wiener|Q.?Wiener|regularity.structure|Hairer|mild.solution|Karhunen.?Loève|Galerkin|Allen.?Cahn|Cahn.?Hilliard|Phi.?4|space.time.noise|spde|SPDE` against `*.go` returns **no callable matches at all** (only the Wiener *filter* in `audio/separation/wiener.go` which is the unrelated Wiener-Hopf signal-processing algorithm). The closest substrate is the deterministic-ODE engine `chaos/ode.go` (RK4 + Euler, no PDE-discretisation surface), the FFT machinery `signal/fft.go` (real DFT for spectral methods on torus), `linalg/decompose.go` (LU/QR/Cholesky/Eigen — would feed Q-Wiener spectral expansion), and the `prob.SampleNormal` Box-Muller keystone shared by **NINETEEN+ Block-C reviews** (still absent at v0.10.0 — gates every space-time-noise discretisation). There is no `pde/` package, no finite-element substrate, no Galerkin projection, no spatial-discretisation primitive whatsoever. **Slot 244 (new-pde-solvers) is the upstream prerequisite for almost everything in this review** — without a 1D/2D Laplacian-FD substrate, even the simplest stochastic-heat solver is built from scratch.
**Summary line 2.** Fifteen SPDE primitives **P1-P15** totalling ~3,200 LOC organized as **(0) substrate ~430 LOC** (P0a `prob/random.go` Box-Muller — shared keystone, P0b `prob/qmc.go` Sobol — shared with 202/263, P0c `pde/laplacian1d.go` — gated by 244 or shipped here as inline ~80 LOC), **(1) Tier-1 keystone ~1,150 LOC** (P1 SpaceTimeWhiteNoise + P2 StochasticHeatEquation-mild-solution + P3 StochasticAllenCahn-Galerkin + P4 KPZ-CoxMatetski-2024-numerical), **(2) Tier-2 high-demand ~880 LOC** (P5 StochasticBurgers + P6 StochasticCahnHilliard + P7 QWienerProcess-spectral + P8 KarhunenLoeveExpansion), **(3) Tier-3 frontier ~740 LOC** (P9 StochasticNavierStokes-2D + P10 RegularityStructures-toy-Phi4_3 + P11 RoughPath-SPDE-coupling cross-link 218 + P12 RandomAttractor + P13 LyapunovExponent-SPDE + P14 Multiscale-SPDE-homogenization + P15 Inviscid-Limit). **CANDOR**: SPDE numerics is the **most niche corner of cutting-edge applied math** the 400-roster covers — outside academic mathematical-physics, quantum-field-theory simulation, and stochastic-fluid-mechanics research, there are **near-zero direct consumers** in reality's downstream stack (no aicore service uses KPZ / Phi-4 / 2D-NSE today). The 2014 Fields-Medal-winning regularity-structures theory (Hairer 2014) gives reality a **prestigious-completeness deliverable**, but the implementation surface is enormous (Hairer's BPHZ-renormalisation algorithm is ~5000 LOC in the Cambridge Mathematica reference and is non-trivial-to-port to zero-dep Go) and the consumer is theoretical-physics not production-AI. **Realistic scoping**: ship Tier-1 (P1+P2+P3+P4 ~1,150 LOC) as the credible "reality has SPDEs" surface; defer Tier-3 entirely until a research-physics consumer pulls. Cheapest-1-day shippable: **P1 SpaceTimeWhiteNoise + P2 StochasticHeatEquation-Galerkin ~280 LOC** on top of `signal/fft.go` Fourier basis + Box-Muller keystone (shared blocker), saturating an R-MILD-SOLUTION-PIN 3/3 against the closed-form Gaussian stationary distribution `u_∞ ~ N(0, (-Δ)^{-1}/2)` for additive-noise stochastic-heat on `[0, 2π]` torus. Singular-cutting-edge piece: **P4 KPZ-equation numerical solver** Bertini-Giacomin-1997 / Cox-Matetski-2024 — KPZ is the canonical *ill-posed-without-renormalisation* SPDE and its solution theory is the entire reason Hairer won the 2014 Fields Medal; ships as a regularised-mollifier-cutoff Cole-Hopf transformation `u = (2ν/λ) log w` reducing KPZ to multiplicative-noise stochastic-heat (`∂w/∂t = ν Δw + (λ/2ν) w·W`). **18 of 15 primitives unique to this slot** (P11 cross-links 218 rough-paths, P3+P6+P7+P8 partially overlap 202 Brownian-path / SDE-spectral-noise discussion). Recommended placement **NEW package `prob/spde/`** ~2,500 LOC (P1-P10) + new substrate `pde/` package ~700 LOC (P0c laplacian1d / laplacian2d) co-shipped with slot 244-new-pde-solvers; rationale: SPDEs need both stochastic-noise primitives (`prob/`) and spatial-PDE-discretisation (`pde/`), so a sub-package of `prob/` keeps the noise-discretisation co-located while consuming the deterministic spatial substrate from `pde/`.

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Repo-wide audit for SPDE / space-time noise / KPZ / regularity-structure / mild-solution / cylindrical-Wiener / Q-Wiener / Karhunen-Loève / Galerkin / stochastic-heat / stochastic-Burgers / Allen-Cahn / Cahn-Hilliard / Φ⁴ / 2D-NSE surface — **zero callable matches** anywhere in `*.go` files outside review-corpus. The full audit:

| Surface | Path | SPDE relevance |
|---|---|---|
| `audio/separation/wiener.go` | `audio/separation/wiener.go` | Wiener *filter* (Wiener-Hopf signal-processing); **NOT Wiener process** — name-collision only |
| `chaos.RK4Step / EulerStep / SolveODE` | `chaos/ode.go` | Deterministic-ODE substrate; gates time-stepping for method-of-lines SPDE discretisation but **no PDE spatial substrate** |
| `signal.FFT / IFFT` | `signal/fft.go` | Real DFT, in-place, power-of-2 lengths; **the only spectral-basis substrate in repo** — gates P7 Q-Wiener spectral expansion + P2 Galerkin-Fourier on torus + P4 KPZ-spectral solver |
| `linalg.QRDecompose / Cholesky / Eigen` | `linalg/decompose.go,eigen.go` | Spectral decomposition for P8 Karhunen-Loève (eigenvalues of covariance operator) + P3 Galerkin mass / stiffness matrices; **PRESENT** |
| `linalg.SolveLinearSystem` | `linalg/matrix.go` | Tridiagonal-banded-solver for FD-Laplacian implicit timesteps; **PRESENT (general LU)** |
| `prob.SampleNormal` Box-Muller | -- | **ABSENT** — keystone blocker shared with 117/184/188/202/216/217/218/219/220/239/240/241 (now twelve+ slots) |
| `prob.MultivariateNormal` | -- | **ABSENT** — needed for spatial-noise covariance sampling |
| `prob/copula` `prob/conformal` sub-packages | `prob/copula/`, `prob/conformal/` | Architectural-template precedent for `prob/spde/` sub-package |
| `chaos/systems` Lorenz Van-der-Pol | `chaos/systems.go` | Deterministic-chaos test cases; orthogonal to SPDE |
| `pde/` package | -- | **ABSENT** — slot 244 will create; SPDE consumes |
| FEM / FE-mass / FE-stiffness | -- | **ABSENT** — slot 247 (mortar-FEM) will create |
| Spectral-methods (Chebyshev/Fourier/RBF) | -- | **PARTIAL** (`signal/fft.go` Fourier only) — slot 245 will create general spectral substrate |
| Multigrid | -- | **ABSENT** — slot 248 will create |
| P1-P15 SPDE primitives | -- | **ALL ABSENT** (15 distinct primitives) |

**Cross-import edges.** `prob/spde/` would need `signal.FFT` (PRESENT) for spectral Q-Wiener expansion, `linalg.Eigen` (PRESENT) for Karhunen-Loève of covariance operator, `linalg.SolveLinearSystem` (PRESENT) for implicit-timestep linear solves on tridiagonal Laplacian, `prob.SampleNormal` (Box-Muller — **ABSENT**) for noise increments, and a `pde.Laplacian1D` substrate (slot 244 — **ABSENT**, but minimally inlinable here at ~80 LOC). New edges: `prob/spde → signal/fft`, `prob/spde → linalg`, `prob/spde → prob` (Box-Muller), `prob/spde → pde` (slot 244 substrate), `prob/spde → chaos` (RK4 / Euler for time-stepping). Cross-link to slot 202 SDE substrate is **light** — SPDEs are not built on SDE solvers (different state-spaces), but slot 202 S1 BrownianPath ships the 1-D Brownian-motion increment generator that SPDE noise discretisation reuses for each spatial mode independently. Cross-link to slot 218 RoughPaths is the rough-path-lift of Q-Wiener (P11 is the dedicated cross-link entry). Cross-link to slot 219 MFG is **light** — McKean-Vlasov SPDE on the master-equation lift, but reality won't ship that surface.

**Cross-package blockers.**

| Blocker | Owner | Blocks | LOC est |
|---|---|---|---:|
| `prob.SampleNormal` Box-Muller | 117/184/188/202/216/217/218/219/220/239/240/241/this-slot-242 | All 15 P-primitives | 50 |
| Sobol / Halton low-discrepancy (P0b) | 202-S0b / 263-Quasi-MC | P14 Multiscale-SPDE QMC-pair, P12 RandomAttractor | 150 |
| 1D Laplacian FD-stencil | 244-new-pde-solvers OR ship-here | P2, P3, P5, P6 | 80 |
| 2D Laplacian FD-stencil | 244-new-pde-solvers OR ship-here | P9 2D-NSE, P10 Phi4_3 | 200 |
| Spectral 2D / 3D FFT | 245-new-spectral-methods | P9 2D-NSE, P10 Phi4_3 | 250 (slot 245) |
| Adaptive ODE timestepper (RK45) | 027-chaos-missing | P5, P9 (stiff regimes) | 220 (slot 027) |

Total upstream-blocker LOC ~880 of which ~50 (Box-Muller) is already four-times-redundant in the Block-C roster (slot 117 ships it as 1-day delivery).

**Cross-link audit.**
- 202-new-sde: **strict upstream** — Brownian-path generation per spatial-mode is reused; SDE strong-order theory does NOT carry over to SPDEs (mild-solution theory is structurally different).
- 218-new-rough-paths: **cross-link** — rough-path-lift of cylindrical-Wiener gives signature-based SPDE solution theory (Hairer 2014 BPHZ uses signature-trees as model space).
- 219-new-mean-field-games: **light cross-link** — McKean-Vlasov SPDE on the master-equation lift; reality won't ship.
- 168-synergy-physics-autodiff: **orthogonal** — adjoint-SPDE for inverse problems is a deferral.
- 244-new-pde-solvers: **strict upstream** for P2/P3/P5/P6/P9/P10; slot 244 ships the 1D/2D/3D Laplacian + FE-mass/stiffness substrate this review consumes.
- 245-new-spectral-methods: **strict upstream** for P4 / P9 / P10 spectral discretisations.
- 247-new-mortar-fem: **upstream** for P3 stochastic-Allen-Cahn FE-Galerkin; can defer FE entirely and ship spectral-Galerkin only.
- 248-new-multigrid: **optional upstream** — multigrid for elliptic SPDE inverse problems; defer.
- 263-new-quasi-mc + 264-new-mlmc: **cross-link** — MLMC-SPDE (Cliffe-Giles-Scheichl-Teckentrup-2011) is the production-standard for SPDE expectations, requires MLMC substrate.

---

## 1. The fifteen SPDE primitives

Each entry: capability + reference / composition / LOC / cross-link / blocking-flag.

### Tier-0 — substrate (~430 LOC)

**P0a — `prob/random.go` Box-Muller / Marsaglia-Polar / Ziggurat ~50 LOC.** Shared keystone with 117/184/188/202/216/217/218/219/220/239/240/241. **TWELFTH-time-flagged** at this point in the Block-C roster. Single 1-day commit unblocks every B-roster downstream including all 15 P-primitives in this review.

**P0b — `prob/qmc.go` Sobol / Halton ~150 LOC.** Shared with slot 202-S0b and slot 263. Sobol direction-numbers (Joe-Kuo 2008 for d ≤ 21201). Brownian-bridge KL-pairing for QMC-SPDE noise generation. Defer until slot 263 ships.

**P0c — `pde/laplacian1d.go` 1D-FD-Laplacian + tridiagonal-implicit-step ~80 LOC.** Either ship in slot 244 or inline here. Standard 3-point stencil `(u_{i+1} − 2u_i + u_{i-1})/Δx²` with Dirichlet / Neumann / periodic boundaries; backward-Euler timestep `(I − Δt/Δx² · L)·u^{n+1} = u^n + noise` solved via `linalg.SolveLinearSystem` (already tridiagonal). Pin against closed-form heat-equation `u(t, x) = e^{-π²t} sin(πx)` to 1e-10. **Required for P2-P3-P5-P6.**

### Tier-1 — keystone (~1,150 LOC)

**P1 — `SpaceTimeWhiteNoise(grid []float64, dt, dx float64, rng RNG) [][]float64` ~80 LOC.** Discretised space-time white noise `dW(t, x) ≈ √(Δt·Δx) · ξ_{i,n}` where `ξ_{i,n} ~ N(0, 1)` iid over space-time grid. Variance `√(Δt/Δx)` for Itô-isometry-correct discretisation (Walsh-1986 Lecture-Notes-on-Stochastic-PDE Chap. 3). Each spatial cell receives an independent N(0,1) draw scaled by `1/√(Δx)` (cell-density normalisation) and the time-step inherits the standard `√Δt` Brownian-increment factor. **Foundational for P2-P5.** Pin: empirical covariance over 10⁵ paths matches `Cov[W(t,x), W(s,y)] = min(t,s)·δ_{xy}/Δx` to 5%. Reference: Walsh-1986 §2, Lord-Powell-Shardlow-2014 *Introduction-to-Computational-Stochastic-PDE* Chap. 5.

**P2 — `StochasticHeatEquation(u0 []float64, dt, dx, T, sigma float64, bc BoundaryCondition, rng RNG) [][]float64` ~250 LOC.** Solves `∂u/∂t = ν Δu + σ dW(t, x)` on `[0, L]` with periodic / Dirichlet / Neumann BC. **Spectral-Galerkin-Fourier method**: project onto first N Fourier modes `e_k(x) = √(2/L) sin(kπx/L)`, the stochastic-heat-equation diagonalises to N independent OU processes `dû_k = -νk²û_k dt + σ dB_k` each solved via the **exact OU transition** (slot 202-S7) `û_k(t+Δt) = û_k(t)·e^{-νk²Δt} + √(σ²(1-e^{-2νk²Δt})/(2νk²))·Z_k`. Reconstruct `u(x, t) = Σ_k û_k(t) e_k(x)` via inverse-FFT. **Mild-solution variation-of-constants formula**: `u(t, x) = ∫_0^t S(t-s) σ dW(s)` where `S(t) = e^{tνΔ}` is the heat semigroup; spectral Galerkin diagonalises S into `e^{-νk²t}` exactly. ~250 LOC. Pin: for additive noise on torus the stationary distribution is `u_∞ ~ N(0, σ²/(2ν) · (-Δ)^{-1})` with mode variances `σ²/(2νk²)`; saturates R-MILD-SOLUTION 3/3 (mode-by-mode variance + spatial-correlation `E[u(x)u(y)]` matches Green's function + L²-energy `E[||u_∞||²] = Σ σ²/(2νk²)` matches closed-form). Reference: Da Prato-Zabczyk-1992 *Stochastic-Equations-in-Infinite-Dimensions* §5.4, Walsh-1986 §3, Lord-Powell-Shardlow-2014 §10.

**P3 — `StochasticAllenCahn(u0, dt, dx, T, eps, sigma, rng) [][]float64` ~280 LOC.** Solves `∂u/∂t = ε² Δu + (u - u³) + σ dW(t, x)` on `[0, L]` periodic — the canonical reaction-diffusion SPDE with bistable potential `V(u) = (1-u²)²/4`, models phase-separation / pattern-formation driven by space-time white noise. **Operator-splitting**: split into linear-stochastic-heat sub-step (P2) + nonlinear-reaction sub-step solved via explicit-Euler `u^{n+1} = u^n + Δt(u^n - (u^n)³)`. Strang-splitting `L^{Δt/2} N^{Δt} L^{Δt/2}` for second-order accuracy. ~280 LOC. Pin: with σ=0 recovers deterministic-Allen-Cahn travelling-front `u(x, t) = tanh((x - ct)/(ε√2))` to 1e-6 over 1000 timesteps; with σ>0 + small ε measure interface-fluctuation-spectrum vs Funaki-1995 sharp-interface limit. Reference: Funaki-1995-AnnProb, Allen-Cahn-1979-ActaMet, Lord-Powell-Shardlow-2014 §10.7.

**P4 — `KPZEquation(h0, dt, dx, T, nu, lambda, sigma, rng) [][]float64` ~340 LOC** ⭐ **CUTTING-EDGE SINGULAR PIECE.** Solves the **Kardar-Parisi-Zhang equation** `∂h/∂t = ν Δh + (λ/2)|∇h|² + σ dW(t, x)` (Kardar-Parisi-Zhang-1986-PRL) — the canonical **ill-posed-SPDE** whose solution theory motivated Hairer's 2014 Fields-Medal regularity-structures theory (Hairer-2013-Inventiones / Hairer-2014-Annals-of-Math). Direct discretisation diverges as Δx → 0; ship the **Cole-Hopf transformation** `Z(t, x) = exp(λ/(2ν) · h(t, x))` which **converts KPZ to the multiplicative-noise stochastic-heat equation** `∂Z/∂t = ν ΔZ + (λσ/(2ν)) Z·dW(t, x)` (Bertini-Giacomin-1997-CommMathPhys). Solve the latter via P2 + Itô-Stratonovich correction (Da Prato-Debussche-2002), recover `h = (2ν/λ) log Z`. ~340 LOC. Pin against the **KPZ universality class scaling exponents** `<(h(t,x) − h(0,x))²> ∼ t^{2/3}` (Family-Vicsek-1985 height-fluctuation scaling) — saturates an R-KPZ-UNIVERSALITY 2/3 (β=1/3 growth-exponent + α=1/2 roughness-exponent + dynamical-exponent z=3/2). Reference: Kardar-Parisi-Zhang-1986-PRL-56, Bertini-Giacomin-1997-CommMathPhys-183, Hairer-2013-Inventiones-198 *Solving-the-KPZ-equation*, Hairer-2014-Annals-of-Math-180 *A-theory-of-regularity-structures*, Cox-Matetski-2024-arXiv-2403 *Convergence-rates-of-KPZ-numerical-schemes*. **The single most prestigious mathematical-physics deliverable in the entire 400-roster** — KPZ is the reason Hairer won the Fields Medal.

**P0c — `pde/laplacian1d.go`** counted under Tier-0 above (~80 LOC) but functionally Tier-1 since P2-P3-P5-P6 all consume it.

### Tier-2 — high-demand (~880 LOC)

**P5 — `StochasticBurgers(u0, dt, dx, T, nu, sigma, rng) [][]float64` ~220 LOC.** Solves `∂u/∂t = ν Δu - u·∂u/∂x + σ dW(t, x)` on `[0, 2π]` periodic — viscous-Burgers with stochastic forcing, **inviscid limit ν→0 connects to Burgers turbulence and the scalar conservation-law theory** (E-Khanin-Mazel-Sinai-2000-AnnMath gives the variational formula for solutions). Operator-splitting: linear-stochastic-heat sub-step (P2) + nonlinear-Burgers sub-step `u_t + u·u_x = 0` solved via Godunov / upwind FD. ~220 LOC. Pin: stationary energy-spectrum `E(k) ∼ k^{-2}` (Kraichnan-Burgers cascade) at high k. Reference: Da Prato-Debussche-Temam-1994-NoDEA, E-Khanin-Mazel-Sinai-2000-AnnMath-151.

**P6 — `StochasticCahnHilliard(u0, dt, dx, T, eps, sigma, rng) [][]float64` ~240 LOC.** Solves the **conserved** order-parameter SPDE `∂u/∂t = Δ(-ε² Δu + u³ - u) + σ ∇·dW(t, x)` — fourth-order operator with conservation `∫u dx = const`. Discretise via spectral-Galerkin (4th derivative diagonalises in Fourier basis as `k⁴`); operator-splitting with implicit linear-step (k⁴ stiffness needs implicit) + explicit nonlinear-step. ~240 LOC. Pin: conservation `||∫u dx|| < 1e-10` per-step + with σ=0 recover deterministic spinodal-decomposition coarsening rate `L(t) ∼ t^{1/3}` (Cahn-Hilliard-1958). Reference: Cardon-Weber-2010-StochProcAppl, Da Prato-Debussche-1996.

**P7 — `QWienerProcess(eigenvalues []float64, eigenfunctions [][]float64, dt, T, rng) [][]float64` ~220 LOC.** Q-Wiener process construction: given a positive trace-class covariance operator Q with eigenvalues λ_k > 0 and eigenfunctions e_k, sample `W^Q(t) = Σ_k √λ_k · β_k(t) · e_k` where β_k are iid standard Brownian motions (Da Prato-Zabczyk-1992 Theorem 4.3). Truncated at first N modes with truncation-error `Σ_{k>N} λ_k`. Cylindrical-Wiener limit: λ_k = 1 for all k (non-trace-class — only well-defined on rigged Hilbert space `H_{-1}`). ~220 LOC. Pin: empirical-covariance `E[<W^Q(t), φ><W^Q(t), ψ>] = t·<Qφ, ψ>` to 5% over 10⁵ paths. Reference: Da Prato-Zabczyk-1992 §4.

**P8 — `KarhunenLoeveExpansion(covarianceKernel func(x, y float64) float64, domain []float64, n int) (eigenvalues, eigenfunctions)` ~200 LOC.** Karhunen-Loève expansion: given covariance kernel C(x, y), discretise on grid, form covariance matrix C_{ij} = C(x_i, x_j), diagonalise via `linalg.Eigen` → eigenvalue / eigenfunction pairs. Truncate at first N modes for noise expansion. Used in P7 (Q-Wiener via KL) and P14 (multiscale-SPDE KL of random coefficient field). ~200 LOC. Pin: for exponential-kernel `C(x,y) = exp(-|x-y|/L)` recover analytic eigenvalues `λ_k = 2L/(1+L²ω_k²)` with ω_k roots of `tan(ωT) = 2Lω/(L²ω²-1)` (Spanos-Ghanem-1989). Reference: Karhunen-1947-AnnAcadSciFenn-A1, Loève-1948.

### Tier-3 — frontier (~740 LOC)

**P9 — `StochasticNavierStokes2D(u0, dt, dx, T, nu, sigma, rng) [][]float64` ~280 LOC ⊘ DEFER.** 2D-NSE with stochastic forcing `∂u/∂t + (u·∇)u = ν Δu - ∇p + σ dW(t, x)`, `∇·u = 0`. Spectral-Galerkin in vorticity formulation `∂ω/∂t = ν Δω - u·∇ω + σ curl(dW)` on `[0, 2π]²` torus. Stochastic 2D-NSE has unique solutions / ergodic invariant measure (Hairer-Mattingly-2006-Annals-of-Math) — **the second 2014-Fields-Medal-flavoured SPDE** in this review (Hairer's first major result before regularity-structures). Defer: 2D spectral-FFT substrate is large (~250 LOC slot 245), and consumer is research-fluid-mechanics not production-AI. Reference: Hairer-Mattingly-2006-Annals-of-Math-164 *Ergodicity-of-the-2D-Navier-Stokes-equation-with-degenerate-stochastic-forcing*, Flandoli-2008-StochProcAppl.

**P10 — `Phi4_3RegularityStructures(...)` ~280 LOC ⊘ DEFER.** Φ⁴_3 quantum field theory SPDE `∂φ/∂t = Δφ - φ³ + ξ(t, x)` on 3D-torus where ξ is space-time-white-noise. **Ill-posed** — divergent renormalisation needed. Hairer-2014 BPHZ-renormalisation gives well-posed theory; numerics via Mourrat-Weber-2017-CommMathPhys mollifier-cutoff `ξ_ε = ξ * ρ_ε` + counterterm `C_ε φ` with `C_ε ~ -ε^{-1} - log(ε)`. Implementation is **enormous** (Hairer's BPHZ algorithm is 5000+ LOC in Mathematica reference). Ship a **2D-version** Φ⁴_2 (mass-renormalisation only, no log-divergence) at ~280 LOC as proof-of-concept. **Defer Φ⁴_3 entirely** — consumer is theoretical-physics, not reality's downstream stack. Reference: Hairer-2014-Annals-of-Math-180, Mourrat-Weber-2017-CommMathPhys-356, Catellier-Chouk-2018-AnnProb-46.

**P11 — `RoughPathSPDECoupling(...)` ~120 LOC ⊘ DEFER.** Cross-link to slot 218-rough-paths. Cylindrical-Wiener has rough-path-lift via Lévy-area construction; SPDEs admit rough-path-pathwise solution theory (Gubinelli-Imkeller-Perkowski-2015-Forum-Math.Pi *Paracontrolled-distributions* gives the alternative to regularity-structures). Defer until 218 rough-path substrate ships and consumer pulls. Reference: Gubinelli-Imkeller-Perkowski-2015 / Catellier-Chouk-2018.

**P12 — `RandomAttractor(spde SPDE, T float64) Attractor` ~60 LOC ⊘ DEFER.** Pull-back attractor `A(ω) = ∩_{T>0} ∪_{t≥T} S(0, -t, ω)·B` for stochastic-dissipative-systems (Crauel-Flandoli-1994-ProbTheoryRel). Numerical pull-back via long-trajectory simulation backwards-in-time. Defer.

**P13 — `LyapunovExponentSPDE(spde SPDE, T float64) []float64` ~100 LOC ⊘ DEFER.** Lyapunov spectrum of SPDE — analogous to chaos/lyapunov but for infinite-dimensional state. Defer until consumer pulls.

**P14 — `MultiscaleSPDE(slowSPDE, fastSPDE, scaleSep float64) HomogenizedSPDE` ~120 LOC ⊘ DEFER.** Stochastic homogenisation: derive effective SPDE for slow variable when fast variable is mixing (Pavliotis-Stuart-2008 *Multiscale-Methods*). Defer.

**P15 — `InviscidLimit(spde SPDE, viscositySchedule func(t) float64) Solution` ~60 LOC ⊘ DEFER.** Inviscid limit ν → 0 of stochastic-Burgers / stochastic-NSE; viscosity-solution theory in stochastic setting. Defer.

---

## 2. Connective tissue — what each new edge buys

| Edge | LOC of glue | What it unlocks |
|---|---|---|
| `prob/spde/ → signal/fft.go` | 0 — already callable | Spectral-Galerkin diagonalisation of Laplacian (P2-P4) |
| `prob/spde/ → linalg.Eigen` | 0 — already callable | Karhunen-Loève of covariance operator (P8) |
| `prob/spde/ → linalg.SolveLinearSystem` | 0 — already callable | Tridiagonal implicit-Euler for stiff regimes (P3 Allen-Cahn small-ε) |
| `prob/spde/ → prob.SampleNormal` | 0 (after Box-Muller lands) | Space-time white-noise sampling (P1) |
| `prob/spde/ → pde/laplacian1d` | 0 — slot 244 substrate | FD-Laplacian for non-spectral methods (P3, P5, P6) |
| `prob/spde/ → chaos/RK4Step` | 0 — already callable | Method-of-lines time-stepping for explicit schemes |
| `prob/spde/ → prob/diffusion/` (slot 241) | 0 — DDPM uses SPDE-like noise | DDPM-on-images naturally connects to spatial noise (research-grade) |
| `prob/spde/ → optim/transport/sinkhorn.go` | 0 — already callable | W_2-distance between SPDE distributions (research-grade) |

Two new packages: `prob/spde/` (~2,500 LOC if Tier-1+Tier-2 ship) and a tiny `pde/` substrate (~700 LOC if 244 ships first). No existing API breaks.

---

## 3. Three architectural recommendations

**F1. Co-ship slot 244-new-pde-solvers and slot 242-new-spde as a paired delivery.** Without a `pde/` substrate (slot 244 owns: 1D/2D Laplacian FD, BC handling, FE-mass-matrix, FE-stiffness-matrix), every SPDE primitive in this review reimplements the spatial substrate from scratch. Ship `pde/laplacian1d.go` (~80 LOC) as the minimal MVP in either slot — both reviews agree this is the gating substrate. The pure-Galerkin-Fourier subset of SPDEs (P2 stochastic-heat on torus) does NOT need `pde/` — only `signal/fft.go` — so P2 ships as 1-day MVP standalone.

**F2. Establish `spde.Scheme` enum mirroring `sde.SDEScheme` (slot 202 F2).**

```go
type SPDEScheme int
const (
    SpectralGalerkinFourier SPDEScheme = iota  // P2 / P4 via FFT
    FiniteDifferenceImplicit                   // P3 / P5 / P6 via pde.Laplacian
    OperatorSplitting                          // P3 / P5 / P6 via Strang
    ColeHopfTransform                          // P4 via Bertini-Giacomin
)
```

Mirrors slot 202's `SDEScheme` clean-result idiom. Avoids API explosion.

**F3. Pin convergence claims via golden-files at multiple Δt and Δx.** SPDE convergence theory has TWO axes: temporal-rate (typically O(Δt^{1/2-ε}) for Euler-Maruyama-in-time) and spatial-rate (typically O(Δx²) for FD-Laplacian, O(N^{-r}) for Galerkin truncation at first N modes). Ship golden-file tests pinning the **product** convergence in (Δt, Δx, N) jointly — e.g. for P2 Stochastic-Heat: `E[||u^{Δt,Δx,N}_T - u^{exact}_T||²] ≤ C·(Δt + Δx² + N^{-2r})` for u₀ ∈ H^r. Reference: Lord-Powell-Shardlow-2014 §10.3 / §10.5 / §10.7 are the canonical cross-language pinning targets.

---

## 4. Risks and gotchas

- **G1. KPZ ill-posedness.** Direct discretisation of KPZ diverges as Δx → 0 — the `(λ/2)|∇h|²` term has Sobolev-regularity below the noise. **Cole-Hopf-transform is the canonical fix** (Bertini-Giacomin-1997) but only works in 1D + multiplicative-noise regime. Hairer-2013/2014 BPHZ is the universal fix but is enormous to implement. Ship Cole-Hopf as the production path; ship BPHZ as research-mode-only behind an experimental-API tag.
- **G2. Cylindrical-Wiener vs Q-Wiener.** Cylindrical-Wiener (Q = I, all eigenvalues 1) is **non-trace-class** — only well-defined on rigged Hilbert space `H_{-1}`. Direct sampling diverges as N → ∞. Ship Q-Wiener (P7) with **explicit decay condition** `Σ λ_k < ∞` validated at construction time; cylindrical-Wiener only available on torus where the dual-pairing is well-defined.
- **G3. Stiffness of high-frequency modes.** Mode k of stochastic-heat decays at rate `νk²` — for νΔt·k² > 2 explicit-Euler is unstable. Ship implicit-time-stepping by default; provide explicit option only for educational comparison.
- **G4. Phi^4_2 vs Phi^4_3.** Φ⁴_2 needs only mass-renormalisation (`C_ε ~ -log ε`); Φ⁴_3 needs additional log-divergent counterterm and is **the** flagship Hairer-2014 deliverable — but the implementation surface is 10× larger. Ship Φ⁴_2 in Tier-3; defer Φ⁴_3 entirely.
- **G5. Operator-splitting strang vs lie.** For P3 Stochastic-Allen-Cahn / P5 Stochastic-Burgers / P6 Stochastic-Cahn-Hilliard, naive Lie-splitting `L^{Δt} N^{Δt}` is order-1 in Δt; Strang-splitting `L^{Δt/2} N^{Δt} L^{Δt/2}` is order-2 but doubles the linear-step cost. Ship Strang as default, Lie as opt-in for diagnostic comparison.
- **G6. Spectral-truncation vs FD-aliasing.** Spectral-Galerkin handles the linear-Laplacian exactly but the nonlinear-term `u·∇u` (P5 Burgers) or `u³` (P3/P6) requires de-aliasing via 2/3-rule (Orszag-1971). Ship 2/3-rule de-aliasing in spectral-method. Cross-link to slot 245 spectral-methods substrate.
- **G7. Path-wise vs probabilistic SPDE.** Hairer's regularity-structures theory delivers **deterministic / path-wise** solutions to ill-posed SPDEs (sample-by-sample); stochastic-process-theory (Da Prato-Zabczyk) delivers **probabilistic / distributional** solutions (in expectation / law). Ship both interpretations clearly labelled — they coincide for well-posed SPDEs but diverge in the regime where renormalisation matters.

---

## 5. Cross-language parity targets

Six pinned tests covering the Tier-1 surface:

| Test | Pin | Tolerance | Reference |
|---|---|---|---|
| `TestSpaceTimeWhiteNoise_Covariance` | empirical Cov over 10⁵ paths matches `δ_{xy}/Δx` | 5% | Walsh-1986 §2 |
| `TestStochasticHeat_StationaryDistribution_Torus` | mode variances `σ²/(2νk²)` match closed-form | 1e-3 | Da Prato-Zabczyk-1992 §5.4 |
| `TestStochasticHeat_MildSolution_VsClosedForm` | additive-noise `u(t) = ∫₀^t S(t-s)σdW(s)` modal Itô-isometry | 1e-9 | Lord-Powell-Shardlow-2014 §10 |
| `TestStochasticAllenCahn_DeterministicLimit_TravelingFront` | σ=0 recovers `tanh((x-ct)/(ε√2))` to | 1e-6 | Allen-Cahn-1979 |
| `TestKPZ_UniversalityClass_GrowthExponent` | `<(h(t)−h(0))²> ∝ t^{2/3}` slope ± 0.05 over t ∈ [10, 10³] | 5% on slope | Family-Vicsek-1985 |
| `TestKPZ_ColeHopfConsistency` | KPZ-Cole-Hopf and direct-mollified-KPZ agree on linear-window | 1e-3 | Bertini-Giacomin-1997 |

Reference cross-language implementation: **Lord-Powell-Shardlow-2014** *Introduction-to-Computational-Stochastic-PDE* ships MATLAB code on Cambridge-Press companion site for chapters 10-11; pin Go reference to MATLAB byte-equality on identical RNG seed (Box-Muller / Ziggurat — the same shared keystone).

---

## 6. Verdict

**Ship Tier-0 + Tier-1 (~1,580 LOC over 5 sprints), defer Tier-3:**
- **Sprint 1 (1 day)**: P0a Box-Muller (~50 LOC) + P0c pde.Laplacian1D (~80 LOC) — substrate. Box-Muller is the **twelfth-time-flagged** keystone shared with 117/184/188/202/216/217/218/219/220/239/240/241; THIS-slot makes 13.
- **Sprint 2 (3 days)**: P1 SpaceTimeWhiteNoise (~80 LOC) + P2 StochasticHeatEquation-Galerkin (~250 LOC) — saturates R-MILD-SOLUTION 3/3, the canonical didactic SPDE.
- **Sprint 3 (1 week)**: P3 StochasticAllenCahn-OperatorSplitting (~280 LOC) — phase-separation / pattern-formation; the canonical reaction-diffusion-SPDE.
- **Sprint 4 (1 week)**: P4 KPZEquation-ColeHopf (~340 LOC) ⭐ **CUTTING-EDGE PIECE** — Hairer-2014 Fields-Medal-flavoured deliverable.
- **Sprint 5 (1 week)**: P5 StochasticBurgers (~220 LOC) + P6 StochasticCahnHilliard (~240 LOC) — Tier-2 high-demand classics.
- **Defer Tier-2**: P7 Q-Wiener (220), P8 Karhunen-Loève (200) — ship when consumer pulls.
- **Defer Tier-3 entirely**: P9 2D-NSE, P10 Phi^4, P11 RoughPath-SPDE, P12 RandomAttractor, P13 LyapunovExponent-SPDE, P14 Multiscale, P15 Inviscid — research-grade only.

**CANDOR closing.** SPDE-numerics is **the most niche corner of cutting-edge applied math the 400-roster covers** outside academic mathematical-physics. reality has **near-zero direct SPDE consumers** in its downstream stack today (no aicore / Pistachio / Pulse / Oracle / Muse / Horizon / Sentinel uses KPZ / Stochastic-NSE / Phi^4 today). The 2014-Fields-Medal Hairer-regularity-structures theory is **prestigious-completeness-grade**, not consumer-pulled. Justification for shipping Tier-1 (P1-P4): theoretical-completeness + frontier-capability + cross-link-leverage to slot 202 SDE / slot 218 RoughPath / slot 219 MFG / slot 241 Diffusion-Models. **The single shippable deliverable that justifies this slot is P4 KPZ-Cole-Hopf** — KPZ universality is *the* mathematical-physics result of 2010-2024 (Corwin-2012-RandomMatricesTheoryAppl-1 *The-Kardar-Parisi-Zhang-equation-and-universality-class* Annual-Review-style survey + Hairer-2014-Fields-Medal + Borodin-Corwin-2014-CommPureApplMath-67 + Quastel-Spohn-2015-StochProcAppl-185 + Cox-Matetski-2024-arXiv-2403). reality should ship P4 not because it has consumers but because it has **mathematical-prestige** — and at ~340 LOC of Cole-Hopf-transform composing P2 spectral-Galerkin, the implementation cost is small relative to the brand-value of "reality ships KPZ".

**Singular-highest-leverage 1-day project:** P1+P2 (~330 LOC) on Box-Muller + signal/fft substrate. Saturates an R-MILD-SOLUTION 3/3 pin and lights up the canonical-SPDE textbook example. Mirrors the architectural-template-of-202-S1 (1-day Brownian-path + Euler-Maruyama on `chaos/ode.go`-style API) — same author, same review-week, same composition pattern.

**Singular-cutting-edge piece:** P4 KPZ-equation-Cole-Hopf — Fields-Medal-flavoured, ~340 LOC, no off-the-shelf zero-dep Go (or any-production-language) library ships it. The Cambridge-Lord-Powell-Shardlow-2014 textbook ships MATLAB-research-code; reality would be the **first production-Go implementation of KPZ-numerical-solver** in any open-source library. Brand-value > consumer-value, and that's enough.

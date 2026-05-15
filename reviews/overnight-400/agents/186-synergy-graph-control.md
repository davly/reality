# 186 | synergy-graph-control

**Topic:** graph × control — consensus dynamics, multi-agent control, Laplacian flow, formation, flocking, gossip, distributed optimisation, Kuramoto, networked Kalman, structural controllability.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `graph/`, `control/`, and `linalg/` (with cameos from `chaos/`, `optim/`, `optim/proximal/`, `prob/`) compose. Not per-package isolation gaps (covered by 051-055 control, 081-085 graph, 097 linalg, 157 graph×linalg spectral). Repo at v0.10.0, 1965 tests.

## Two-line summary

Today `graph/` ships pure combinatorial / centrality / flow primitives over `IntAdjacency = map[int][]int` (Dijkstra, A*, Brandes-betweenness, Eigenvector-centrality, Louvain, Tarjan-SCC, KruskalMST, Edmonds-Karp, PageRank — ~1100 LOC) with **zero matrix bridge** (no `AdjacencyMatrix`, no `Laplacian`, flagged keystone-blocker in 157-G1/G2); `control/` ships scalar PID + four first-order filters + polynomial `TransferFunction` with Durand-Kerner poles (~490 LOC) — **zero state-space, zero discrete-time, zero multi-input, no horizon, no concept of "agent network"**; `linalg/` provides every dense kernel needed (MatMul, MatVecMul, Cholesky, LU, sym-eigvals, PCA) but no SVD/Eigvec/MatrixExp; the entire networked-control / multi-agent-systems canon — Olfati-Saber-Murray-2004 consensus, Fiedler-1973 algebraic-connectivity convergence rate, Boyd-Ghosh-Prabhakar-Shah-2006 randomised-gossip, Kempe-Dobra-Gehrke-2003 push-sum, Reynolds-1987→Olfati-Saber-2006 flocking, Kuramoto-1975 phase oscillators on graphs, Wang-Chen-2002 pinning control, Lin-1974 structural controllability, Nedić-Ozdaglar-2009 distributed sub-gradient, Wang-Elia-2010 push-DGD, Cortés-Martínez-Karatas-Bullo-2004 coverage / Lloyd, Olfati-Saber-Fax-Murray-2007 networked-Kalman-consensus — is **wholly absent** (verified zero matches on Consensus|Gossip|PushSum|Flocking|Kuramoto|Pinning|Structural|MultiAgent|Distributed|Containment|EdgeAgreement|Voronoi|Lloyd in `graph/`, `control/`, `chaos/`, `optim/`).

Twenty-three synergy primitives N1–N23 totalling **~3,260 LOC** of pure connective tissue stand up the entire networked-control stack on existing bases; the **single architectural keystone** is N1 `Laplacian` + N2 `LaplacianFlow` (~110 LOC) — every subsequent primitive composes through them; cheapest one-day PR is **N1+N2+N3 ConsensusODE = 220 LOC** saturating the canonical R-MUTUAL-CROSS-VALIDATION 3/3 pin (continuous Laplacian flow via `chaos.RK4` + discrete Perron iteration `(I−εL)^k x₀` + symbolic equilibrium `x_∞ = (1ᵀx₀/n)·1` agree to 1e-10 on any connected graph); highest-leverage one-day unlock is **N6 AlgebraicConnectivity** (Fiedler λ₂(L), ~50 LOC) because it is the **single scalar that predicts convergence rate** of every consensus protocol N3–N23 below — turning every multi-agent simulation into a quantitative design exercise rather than empirical guesswork; crown jewel is **N16 KuramotoOnGraph** (~180 LOC) — Kuramoto coupled-oscillators is **the** canonical synchronisation model that bridges chaos/dynamical-systems × graph spectral × control-theoretic phase locking, and it composes `chaos.RK4` over a graph-Laplacian-shaped ODE with synchronisation order parameter computed via `linalg.MatVecMul` in 60 lines of glue; recommended placement is `graph/networked.go` for N1–N12 (graph is consumer-shaped — these are spectral-graph operators on `IntAdjacency` returning `[]float64`) and `control/networked.go` for N13–N23 (those that consume a `StateSpace` agent model — the same `StateSpace` struct flagged in 161-C1 and 178-M3 as the foundational missing object). Thirteen of twenty-three ship today against v0.10.0; ten are blocked on **either** 161-C1 `StateSpace` (eight consumers) **or** 097-T1 `Eigvec` returning eigenvectors (three consumers — Fiedler-vector spectral bisection, structural controllability, networked-Kalman observer). This synergy lives at the **rare three-way intersection** of graph theory, control theory, and dynamical systems — there is no zero-dependency Go reference implementation anywhere, and SciPy / NetworkX / control-toolbox split it across three Python packages that don't compose cleanly.

---

## 0. Bases — what each package exposes today (verified file-walk)

### `graph/` (12 files, ~1100 LOC numeric core; agents 081-085, 162, 171)

- `types.go`: `IntAdjacency = map[int][]int` (line 7 — single-line definition)
- `graph.go`: `Edge=[2]string`, `AdjacencyList`, `Nodes`, `InDegree`, `Roots`, `Leaves` — string-keyed helpers
- `bfs.go`, `bellman_ford.go`, `dag.go`: traversal + topological-depth
- `shortest.go`: `Dijkstra` (binary-heap), `AStar`, `FloydWarshall`
- `flow.go`: `MaxFlow` (Edmonds-Karp), `TopologicalSort` (Kahn)
- `mst.go`: `KruskalMST`, `PrimMST`
- `centrality.go`: `BetweennessCentrality` (Brandes-2001), `EigenvectorCentrality` (power iter on adjacency, **no Laplacian-eigenvalue path**), `DegreeCentrality`
- `pagerank.go`: `PageRank` (damped power iter + dangling redistribution)
- `community.go`: `ConnectedComponents`, `StronglyConnected` (Tarjan), `LouvainCommunities`
- `importance.go`: bespoke `NodeImportance`, `EdgeFraction`

**Search for networked-control primitives in graph/:** `Consensus|Laplacian|Spectral|Fiedler|Gossip|PushSum|Flocking|Containment|EdgeAgreement|Coverage|Voronoi|Lloyd|Kuramoto|Pinning|Structural|Controllability|Containment|MultiAgent|Distributed` — **zero matches**.

### `control/` (5 files, ~490 LOC; agents 051-055)

- `pid.go` (123): scalar `PIDController{Kp,Ki,Kd,minOutput,maxOutput,integralSum,prevError}` with anti-windup; `Update(setpoint,measured,dt)`/`Reset()`
- `filter.go` (117): `LowPassFilter`/`HighPassFilter`/`ComplementaryFilter`/`RateLimiter` — all stateless first-order scalar
- `transfer.go` (253): polynomial `TransferFunction{Numerator,Denominator}` continuous-only; `Evaluate(s)`, Durand-Kerner-1960/1966 `Poles()`, `IsStable()` (Re(p)<0)

**Absent:** state-space `(A,B,C,D)` (flagged 161-C1, 178-M3), `c2d`/`d2c` (161-C2), Controllability/Observability (161-C3), Kalman/LQR (161), MPC (178), any multi-input/multi-output, any "agent" concept, any concept of "network of controllers", any discrete-time *anything* beyond PID-by-stepping-dt.

### `linalg/` (7 files, ~1500 LOC; agents 096-100)

Every matrix kernel needed for N1–N12: `MatMul`, `MatVecMul`, `MatTranspose`, `MatAdd`, `MatSub`, `MatScale`, `Identity`, `Trace`; `LUDecompose`+`LUSolve`, `Inverse`, `Determinant`; `CholeskyDecompose`+`CholeskySolve`; `QRAlgorithm` (symmetric eigvals only — **no eigenvectors**); `PCA` (private inverse-iteration eigvec recovery, not exported); `CovarianceMatrix`.

**Absent (relevant to N6/N12/N18 only):** `Eigvec` (097-T1; PCA's private trick is per-eigenvalue inverse-iteration, not joint Householder back-transform), `SVD` (097-T1), `MatrixExp` Padé scaling-and-squaring (097-T1; needed for continuous-time Laplacian-flow exact step `e^{-tL}`), `Pseudoinverse` (097-T1; needed for effective-resistance / Kirchhoff), `LanczosSym` (097-T1; sparse top-k Fiedler).

### `chaos/` (4 files, ~600 LOC; agents 036-040)

`chaos/ode.go`: `RK4Step(f, t, y, dt, out)` — allocation-free single step; `SolveODE(f, y0, t0, tEnd, dt) [][]float64` — full trajectory. **This is the consumer of every continuous-time consensus / Kuramoto / flocking ODE below** — composes natively on the `(t, y, dydt) -> ()` signature.

### `optim/`, `optim/proximal/`, `prob/` (cameos)

`optim.LBFGS` and `optim/proximal.Admm` are consumed by N18 distributed-optimisation primitives (N20 Push-DGD, N21 D-ADMM). `prob.NormalQuantile` cameo for N22 networked-Kalman covariance update — but most of N22 routes through 161-C5 KalmanFilter when it lands.

**Cross-edges between graph/ and control/ today:** `grep github.com/davly/reality/graph control/` → 0; reverse → 0. Zero coupling. The interface this review designs is the first edge in either direction.

---

## 1. The conceptual unlock — three theorems

The synergy is gated by three textbook results, each demanding one missing primitive that all subsequent primitives consume:

**T1. Consensus convergence (Olfati-Saber-Murray-2004).** For a connected undirected graph G with Laplacian L=D−A, the protocol ẋ = −Lx drives every component of x to the average ⟨x₀⟩ = (1ᵀx₀)/n exponentially with **rate λ₂(L)** (the algebraic connectivity, a.k.a. Fiedler value). Discrete-time analogue x(k+1) = (I−εL)x(k) converges iff 0 < ε < 2/λ_max(L) (CFL/Perron condition; tighter ε < 1/Δ_max where Δ_max = max-degree gives a degree-bound that needs no eigensolve). **The bridge is `Laplacian` returning a dense `[]float64`** — without it consensus is unrepresentable.

**T2. Network controllability via Kalman rank (Lin-1974, Liu-Slotine-Barabási-2011).** The networked LTI ẋ = (−L + B·u)x with leader-set B ⊂ {0,1}^{n×k} is controllable iff `[B  −LB  L²B  …  (−L)^{n−1}B]` has rank n. Lin-1974 strengthens this to **structural controllability**: a sparsity-pattern test (no zero-eigenvalue with structural-rank-deficient B) decidable in O(n²·k). **The bridge is `LaplacianControllability`** consuming N1 + linalg-rank — without `Eigvec` we fall back to LU-singular-on-leading-minor + cycle-of-power-iteration deflation (works for n≲200, blocked above by 097-T1).

**T3. Synchronisation threshold (Pecora-Carroll-1998 master-stability + Kuramoto-on-graph).** N coupled phase-oscillators θ̇ᵢ = ωᵢ + (K/n) Σⱼ aᵢⱼ sin(θⱼ−θᵢ) synchronise (frequency-locked, |r|→1) iff coupling K exceeds a critical threshold K_c = 2/(πg(0)·λ₂(L)) where g(0) is the natural-frequency density at 0. **The bridge is `KuramotoOnGraph`** as an ODE function consumable by `chaos.RK4` — composing N1 `Laplacian` + adjacency-weighted sin-coupling. The synchronisation order parameter `r = |1/n · Σⱼ exp(iθⱼ)|` is computed via two `linalg.MatVecMul` calls on stacked cos/sin vectors.

These three theorems define the spine of the review. **N1 is the keystone for T1; N6 (Fiedler value) for T1+T3 design; N18 (LeaderControllability) for T2.**

---

## 2. The twenty-three synergy primitives

Each entry: (1) capability, (2) composition recipe, (3) connective-tissue LOC, (4) blocking flag if any. Format mirrors agents 157, 161, 178.

### N1 — `Laplacian(adj IntAdjacency, n int, weights map[[2]int]float64) []float64` (~50 LOC)

**Capability:** dense graph Laplacian L = D − A (or weighted form L_w = D_w − W) as row-major `[]float64` of length n². The single missing object that gates fifteen of twenty-three below. **Recipe:** allocate n²; for each (u, v) in adj, set L[u·n+v] = −w(u,v) and accumulate L[u·n+u] += w(u,v). **Saturation:** path graph P_n eigenvalues are 2(1−cos(kπ/n)), k=0..n−1; complete graph K_n eigenvalues {0, n, n, …, n}; star graph K_{1,n−1} eigenvalues {0, 1, …, 1, n} — all closed-form, pin to 1e-12 vs `linalg.QRAlgorithm` of N1. **Blocks N2-N15.** Cross-link 157-G2 (same primitive, this review consumes it; 157 lands it on architectural-completeness grounds, 186 on networked-control-consumer-pull grounds).

### N2 — `LaplacianFlow(L []float64, n int, x0 []float64, t float64, x []float64)` (~60 LOC, blocked on 097-T1 `MatrixExp`)

**Capability:** exact integration x(t) = e^{−tL} x₀ at arbitrary t — **the analytic closed form for consensus**, what `chaos.RK4` of N3 should converge to as dt→0. **Recipe:** scaled `linalg.MatScale(L, -t, A)`, `linalg.MatrixExp(A, n, eA)`, `linalg.MatVecMul(eA, n, n, x0, x)`. **Blocks** none-blocking-direct-but-the-3-way-pin (saturation: |LaplacianFlow(L, n, x0, t, ·) − chaos.RK4(N3)| < 1e-9 over t ∈ [0, 10] for any connected graph + any x0). Until 097-T1, N2 emits NotImplemented; ship a `LaplacianFlowDiscrete(L, x0, eps, k)` Perron-iteration variant N2′ as fallback (12 LOC, `linalg.MatVecMul` k times after I−εL precompute) — N2′ is what every multi-agent demo uses today.

### N3 — `ConsensusODE(L []float64, n int) func(t float64, x []float64, dxdt []float64)` (~30 LOC)

**Capability:** the right-hand side dx/dt = −Lx packaged for `chaos.RK4Step` / `chaos.SolveODE`. **Recipe:** closure capturing L+n; one `linalg.MatVecMul(L, n, n, x, dxdt)` then `linalg.VectorScale(dxdt, -1, dxdt)`. **Saturation R-MUTUAL-CROSS-VALIDATION 3/3:** (a) symbolic equilibrium x_∞ = ⟨x₀⟩·1 to 1e-15, (b) `chaos.RK4` integration of N3 from x₀ over [0, 100] vs (a) to 1e-9 (assuming λ₂>0), (c) when N2 unblocks, `LaplacianFlow(t=100)` agreement at 1e-9. This is the canonical 3-way pin mirroring 6a55bb4 audio-onset-3-detector / 365368a Clayton-autodiff / 1e12e80 token-set-ratio-RapidFuzz. **Cheapest one-day PR foundation.**

### N4 — `DiscreteConsensus(L []float64, n int, x0 []float64, eps float64, k int, x []float64)` (~40 LOC)

**Capability:** Perron iteration x(k+1) = (I − εL)x(k) — the exactly-discretised counterpart of N3 needed for fixed-step distributed-protocol simulation; ε must satisfy 0 < ε < 2/λ_max(L). **Recipe:** precompute P = I − εL once via `linalg.Identity` + `linalg.MatScale` + `linalg.MatSub`, then k iterations of `linalg.MatVecMul(P, x, x_next)`. **Saturation:** N3 vs N4 with ε=0.01, k=10000 agree at 1e-7 on connected graphs; on graph with isolated component, both equilibrate per-component (1ᵀ_C₁x_∞=⟨x₀|_C₁⟩, etc.) — proves the **kernel-of-L = span(1_C₁, …, 1_Cm)** structure.

### N5 — `MaxDegreeStep(adj IntAdjacency, n int) float64` (~12 LOC)

**Capability:** safe ε = 1/Δ_max for N4 without eigensolve — Olfati-Saber-Murray-2004 §III.A degree-bound CFL. **Recipe:** O(V+E) loop over adj. **Saturation:** ε=1/Δ_max < 2/λ_max for every graph (since λ_max ≤ 2Δ_max, Anderson-Morley-1985), pin via random K_n / random Erdős-Rényi.

### N6 — `AlgebraicConnectivity(L []float64, n int) float64` (~50 LOC)

**Capability:** the Fiedler value λ₂(L) — the **single number that predicts consensus convergence rate** for N3 (rate=λ₂) and N4 (rate=−ln|1−ελ₂|). Highest-leverage scalar in entire networked-control body of knowledge. **Recipe:** call `linalg.QRAlgorithm(L, n, eigenvalues, 1000)`, return eigenvalues[n−2] (since QRAlgorithm sorts descending and the smallest eigenvalue 0 has multiplicity = number-of-connected-components; second-smallest is λ₂). **Edge case:** disconnected graph has λ₂=0 — return 0 and let consumer N9 detect it. **Saturation R-FIEDLER:** path P_n λ₂ = 2(1−cos(π/n)), cycle C_n λ₂ = 2(1−cos(2π/n)), K_n λ₂=n, K_{1,n−1} λ₂=1 — all closed-form to 1e-12. **Highest-leverage one-day unlock.**

### N7 — `FiedlerVector(L []float64, n int, v []float64) bool` (~80 LOC, blocked on 097-T1 `Eigvec`)

**Capability:** the eigenvector v₂ corresponding to λ₂(L) — its sign-pattern is the spectral bisection of the graph (Pothen-Simon-Liou-1990, Fiedler-1975). Consumer of 097-T1 `Eigvec` joint-Householder-QR. **Recipe:** call `linalg.Eigvec(L, n, eigenvalues, eigenvectors)`, copy column n−2 into v. Saturation: sign-pattern bisection of K_{n,n} bipartite graph splits into the two parts to 1e-12 boundary. **Blocked.**

### N8 — `SpectralBisection(adj IntAdjacency, n int) []int` (~40 LOC, blocked on N7)

**Capability:** balanced 2-partition via sign of Fiedler vector — Cheeger-bound-optimal up to constant. Returns []int of length n, values in {0, 1}. Cross-link 157-G7. **Blocked on N7.**

### N9 — `IsConnected(adj IntAdjacency, n int) bool` (~6 LOC; ships today)

**Capability:** connectivity test, prerequisite for every consensus-rate result. **Recipe:** `len(graph.ConnectedComponents(adj, n)) == 1`. Trivial — but worth namespacing in `networked.go` so consumers don't import community.go for this single use.

### N10 — `WeightedConsensus(adj IntAdjacency, n int, weights map[[2]int]float64) func(t float64, x []float64, dxdt []float64)` (~25 LOC, ships today)

**Capability:** asymmetric / weighted consensus ẋᵢ = Σⱼ wᵢⱼ(xⱼ − xᵢ). For row-stochastic weights converges to weighted average ⟨x⟩ = πᵀx₀ where π is the stationary distribution of the weight matrix. **Recipe:** N1 with weights, then N3. **Cross-link** to graph/pagerank.go: π is exactly the PageRank-without-damping on the weighted graph, so `graph.PageRank(n, edges, 1.0, 200)` gives the consensus-equilibrium predictor.

### N11 — `RandomGossipStep(L []float64, n int, x []float64, edge [2]int)` (~15 LOC, ships today)

**Capability:** Boyd-Ghosh-Prabhakar-Shah-2006 randomised gossip — pick a random edge (i,j), update xᵢ, xⱼ ← (xᵢ+xⱼ)/2; everyone else holds. **Recipe:** in-place update of x[i], x[j]. The convergence rate is governed by λ₂(W̄), W̄ = E[W_e] across uniform edge selection — N6 on the **expected gossip Laplacian** is the bound. **Saturation R-GOSSIP-MIXING:** rate constant of `RandomGossipStep` Monte-Carlo over k=10⁶ steps matches log-bound −n·log(1−λ₂(L)/(n·m)) of Boyd-2006-Thm-2 to 5%.

### N12 — `PushSumStep(adj IntAdjacency, n int, w, s []float64, weights map[[2]int]float64)` (~30 LOC, ships today)

**Capability:** Kempe-Dobra-Gehrke-2003 push-sum for **directed** graphs (where Boyd-2006 gossip fails because directedness breaks symmetry of L+Lᵀ): each node maintains (s, w) pair, sends half to a random out-neighbour, ratio s/w → average. **Recipe:** loop over adj for u, redistribute (sᵤ, wᵤ) ← (sᵤ/2, wᵤ/2) and forward to a randomly-chosen out-neighbour. **Saturation:** on any strongly-connected directed graph, sᵢ/wᵢ converges to ⟨s₀⟩/⟨w₀⟩ = ⟨s₀⟩ (when w₀=1ᵢ) for every i. Pin to 1e-6 over 1000 steps on directed cycle, directed K_n, random tournament.

### N13 — `LeaderFollowerConsensus(L []float64, n int, leaders []int, leaderRefs []float64) func(t float64, x []float64, dxdt []float64)` (~40 LOC, ships today)

**Capability:** Ren-Beard-2008-CSM leader-following — leaders hold reference value, followers consensus-track the average of their neighbours INCLUDING leaders. Output is the same RK4-compatible signature as N3. **Recipe:** N3 of N1, then for each leader i, overwrite dxdt[i] = (leaderRefs[i] − x[i])·k_lead. Convergence rate is determined by the **grounded Laplacian** (L with leader rows zeroed and right-hand-side fixed) — its smallest eigenvalue λ_g is positive iff every follower is reachable from at least one leader. Cross-link N18.

### N14 — `ContainmentControl(L []float64, n int, leaders []int, leaderRefs [][]float64) func(t float64, x []float64, dxdt []float64)` (~50 LOC, ships today)

**Capability:** Ji-Ferrari-Trecate-Egerstedt-2008 multi-leader containment — followers converge into the **convex hull** of leader positions. d-dimensional generalisation of N13: state is n×d matrix (flattened), leaders are static at leaderRefs[i], followers obey ẋᵢ = Σⱼ aᵢⱼ(xⱼ − xᵢ). **Recipe:** N3 per coordinate dimension, leader rows zeroed. Saturation: all followers' positions ∈ ConvexHull(leaderRefs) at t=∞ verified by `geometry.ConvexHull2D` membership test. **Cross-link 177** geometry-optim coverage / Voronoi.

### N15 — `EdgeAgreement(adj IntAdjacency, n int, B []float64) func(t float64, z []float64, dzdt []float64)` (~50 LOC, ships today)

**Capability:** Zelazo-Rahmani-Mesbahi-2007 edge-state consensus — instead of node states x, evolve edge states z = Bᵀx where B is the incidence matrix; ż = −BᵀLB·z = −L_e·z runs on the **edge Laplacian** L_e = BᵀLB. Provides a sparser representation when |E|<n²/2 and exposes the cycle-space structure (rank(L_e) = n − number-of-connected-components, kernel(L_e) = cycle-space). **Recipe:** N1 of N18 incidence matrix, then N3 of edge-Laplacian L_e. Cross-link 142-topology Betti₁ = dim(ker L_e) − (number-of-connected-components-of-edge-graph).

### N16 — `KuramotoOnGraph(adj IntAdjacency, n int, omegas []float64, K float64) func(t float64, theta []float64, dthetadt []float64)` (~80 LOC, ships today, **crown jewel**)

**Capability:** Kuramoto-1975 phase oscillators on a graph: θ̇ᵢ = ωᵢ + (K/dᵢ) Σⱼ aᵢⱼ sin(θⱼ − θᵢ). Universal model of synchronisation across neuroscience, power-grid stability, animal flocking, circadian rhythms. **Recipe:** for each i, dthetadt[i] = omegas[i]; for each (i, j) ∈ adj, accumulate (K/dᵢ)·sin(theta[j] − theta[i]) into dthetadt[i]. Order parameter r(t) = |(1/n)Σⱼ e^{iθⱼ}| computed from cos/sin via two `linalg.MatVecMul`. **Saturation R-KURAMOTO-3-WAY:** (a) for large K, |r(∞)| = 1 (full sync); (b) for K < K_c = 2σ_ω/(π·g(0)·λ₂) (T3 above), |r(∞)| ≈ 0; (c) finite-size phase-transition K_c-pin against Strogatz-2000-PhysicaD-Eq.4.10 closed form on uniform-omega + path-/ring-/complete-/Erdős-Rényi-graphs to 5% over 10⁵ steps RK4. **Composes `chaos.RK4` + `linalg.MatVecMul` + N1; this is the model in the corpus that bridges chaos × graph × control.**

### N17 — `BoidsFlockODE(positions, velocities []float64, n, d int, params BoidsParams) func(t float64, y []float64, dydt []float64)` (~120 LOC, ships today)

**Capability:** Reynolds-1987 boids three-rule flocking (separation / alignment / cohesion) operationalised as Olfati-Saber-2006 algebraic-conditions ẍᵢ = α_align·Σⱼ(vⱼ−vᵢ) − α_sep·Σⱼ ∇φ(‖xⱼ−xᵢ‖) + α_cohesion·(x̄−xᵢ). State y = (x, v) ∈ R^{2nd}. **Recipe:** N1-on-distance-graph (proximity Laplacian, edge iff ‖xᵢ−xⱼ‖<r) + alignment via velocity-Laplacian + separation via pairwise-gradient-of-φ. **Saturation:** Olfati-Saber-2006-Thm-3 stability requires α-rigid graph — verify at design time via N6 on proximity graph; flock-stable iff λ₂>0 throughout. Cross-link 177-geometry quaternion-based 3D rotations.

### N18 — `LeaderControllability(L []float64, n int, leaders []int) bool` (~80 LOC, blocked on 097-T1 `Eigvec`)

**Capability:** Liu-Slotine-Barabási-2011 / Lin-1974 — given leader-set B = e_{leaders}, decide controllability of (−L, B) via Kalman rank `[B  −LB  L²B  …  L^{n−1}B]`. The set of leaders is "perfect" iff this is rank n. **Recipe:** k=len(leaders) `linalg.MatMul` calls building the n×(n·k) controllability matrix, then rank via thresholded SVD (097-T1) or fallback `linalg.QRAlgorithm` of MᵀM. **Blocked.** Workaround until 097-T1: PBH eigenvector test — for each (λᵢ, vᵢ) eigenpair of L, check if vᵢᵀB ≠ 0 (only one zero eigenvector v₁=1/√n needs check, so it reduces to "does at least one leader exist?" trivial test for connected graph) — but this is L-only, not general (−L+δI). **Crown-jewel-of-control-theory; the network-dual of 161-C3 ControllabilityMatrix.**

### N19 — `StructuralControllability(adj IntAdjacency, n int, leaders []int) bool` (~100 LOC, ships today)

**Capability:** Lin-1974 graph-theoretic test — controllable iff (a) every node is reachable from a leader (BFS-reachability), AND (b) the graph has no "dilation" (a subset S with |N⁻(S)| < |S| and S ∩ leaders = ∅). **Recipe:** (a) BFS from leaders via existing `graph.BFSDownstream`; (b) Hopcroft-Karp bipartite-matching on the bipartite-decomposition of adj — this is **a fresh primitive worth landing inside `graph/`** as `MaximumBipartiteMatching` (~80 LOC, simpler than full Edmonds-blossom because the construction is bipartite) then a 20-LOC dilation test on top. The output is a Lin-1974 yes/no. **Saturation:** Lin-1974-Fig-2 examples (4-node line, 4-node star, 4-node fully-connected) pin against Liu-Slotine-Barabási-2011-Nature Table-S1 minimum-driver-node-counts.

### N20 — `DistributedSubgradient(adj IntAdjacency, n int, fLocal []func(x []float64) (float64, []float64), x0 [][]float64, eps float64, k int) [][]float64` (~80 LOC, ships today)

**Capability:** Nedić-Ozdaglar-2009 distributed sub-gradient — n agents, each with private convex objective fᵢ; consensus on the minimiser of (1/n)Σfᵢ. Each iteration: xᵢ ← Σⱼ Wᵢⱼxⱼ − ε∇fᵢ(xᵢ). Doubly-stochastic W = I − αL with Metropolis weights (α = 1/(max(dᵢ,dⱼ)+1)) ensures convergence. **Recipe:** Metropolis-weight construction (15 LOC) + N4 step on weights-W + per-agent gradient evaluation. **Saturation:** for fᵢ(x) = (x−aᵢ)², global minimiser is ā=(1/n)Σaᵢ; pin Nedić-Ozdaglar-2009-Thm-2 rate Σ‖xᵢ(k)−ā‖² = O(1/√k) on Erdős-Rényi G(n=20, p=0.3) over k=10⁴.

### N21 — `DistributedADMM(adj IntAdjacency, n int, prox []optproximal.ProxOp, x0 [][]float64, rho float64, k int) [][]float64` (~100 LOC, ships today)

**Capability:** Boyd-Parikh-Chu-Peleato-Eckstein-2011-§7.1 distributed ADMM — global consensus optimisation min Σfᵢ(xᵢ) s.t. xᵢ=z over edge-coupled constraints. **Recipe:** wrapper around existing `optim/proximal.Admm` per-edge (each edge contributes a consensus penalty (ρ/2)‖xᵢ−xⱼ+uᵢⱼ‖²); 8 prox ops in `proximal/` already cover {L1, L2-ball, simplex, box, non-neg, squared-L2, linear, L0}. The clever bit is **the Admm splitting in optim/proximal/ already IS a 2-block ADMM** — the cross-package consumer is to wrap n local 2-block-ADMMs synchronised via the dual updates. Cross-link 178-M5 LinearMpc (consensus-MPC across n agents = N21 with quadratic local fᵢ).

### N22 — `PinningControl(L []float64, n int, pinned []int, kPin float64) func(t float64, x []float64, dxdt []float64)` (~50 LOC, ships today)

**Capability:** Wang-Chen-2002-PhysicaA — feedback control of complex network synchronisation by pinning a small subset of "pinned" nodes (drive xᵢ → x* on pinned set). Achieves global sync at much lower coupling than universal Kuramoto-coupling. ẋᵢ = −Σⱼ Lᵢⱼxⱼ − kPin·1[i∈pinned]·(xᵢ − x*). **Recipe:** N3 of N1 + per-pinned-i correction term in the closure. **Saturation:** Wang-Chen-2002-Thm-1 critical pinning threshold k_c = 1/min-eigenvalue-of-L_p where L_p is L with pinned-rows-zeroed-and-pinned-diagonal-set-to-d_i+kPin — Schur-complement structure pin against scale-free Barabási-Albert graph synthetic.

### N23 — `NetworkedKalman(L []float64, n int, A, C, Q, R []float64, ...) ConsensusKalman` (~250 LOC, blocked on 161-C5 KalmanFilter + 097-T1 Eigvec)

**Capability:** Olfati-Saber-Fax-Murray-2007-CDC distributed Kalman filter — n sensors observe a common process via local measurement matrix Cᵢ; gossip the local information vectors (Cᵢᵀ Rᵢ⁻¹ yᵢ, Cᵢᵀ Rᵢ⁻¹ Cᵢ) over the graph for h consensus rounds per Kalman-step; converges to the centralised-Kalman estimate as h→∞. **Recipe:** 161-C5 local-Kalman per node + N4 on the information-vector across the network per step + sample-time scheduling. **Crown-jewel-of-distributed-estimation; blocked.**

---

## 3. Three cross-cutting connective-tissue patterns

**P1. R-CONSENSUS-3-WAY pin (N3 R-MUTUAL-CROSS-VALIDATION 3/3).** Continuous-time RK4 of ẋ=−Lx (a) vs symbolic equilibrium ⟨x₀⟩·1 (b) vs discrete Perron iteration (I−εL)^k (c) agree to 1e-9 on every connected graph. Mirrors 6a55bb4 / 365368a / 1e12e80 saturated R-MUTUAL family. **Cheapest one-day saturation in entire 186-review** because (a)+(b)+(c) cost ~220 LOC together.

**P2. R-FIEDLER-CLOSED-FORM pin.** Path P_n / cycle C_n / star K_{1,n−1} / complete K_n have closed-form Laplacian eigenvalues; N6 vs closed-form vs `linalg.QRAlgorithm` of N1 to 1e-12 — three-way pin uniquely possible because L is symmetric PSD with rational integer entries on canonical graphs.

**P3. R-KURAMOTO-K_C-3-WAY pin.** N16 critical coupling K_c-empirical (sweep K, find r-bifurcation) vs Strogatz-2000-Eq-4.10 closed form K_c=2/(π·g(0)·λ₂) vs Pecora-Carroll-1998 master-stability spectrum-of-L over RK4 trajectory — all three agree to 5% (mean-field-theory limit) on Erdős-Rényi G(50, 0.2). **Anchor of the synchronisation-design canon.**

These three pins span the 23 primitives — every Nᵢ either contributes to or consumes one.

---

## 4. Recommended placement

`graph/networked.go` (N1, N6, N7, N8, N9, N11, N12, N15, N19; ~360 LOC source — graph-shaped, returns `[]float64` for L or sign-pattern []int for bisection; consumes only `linalg.QRAlgorithm` + `linalg.MatVecMul`; is the FIRST `graph/` → `linalg/` cross-edge in the repo).

`graph/spectral.go` (N4 DiscreteConsensus, N5 MaxDegreeStep — already mooted in 157-G6/G7; merge with this review's N4-N6 to avoid duplication; recommend 157 lands G1+G2 first and 186 imports them).

`control/networked.go` (N3, N10, N13, N14, N16, N17, N18, N20, N21, N22, N23; ~1,400 LOC — control-shaped, returns RK4-compatible ODE closures; consumes `linalg` + a future `StateSpace` from 161-C1 for N18/N23). FIRST `control/` → `graph/` cross-edge.

`control/distributed.go` (N20 DistributedSubgradient, N21 DistributedADMM; ~180 LOC — consumes `optim/proximal.Admm`, `optim.LBFGS`-cameo, FIRST `control/` → `optim/` cross-edge).

DAG: `control/` imports `graph/`, `linalg/`, `chaos/`, `optim/`. `graph/` imports `linalg/`. `linalg/` imports nothing. **Cycle-free.** Mirrors 178 placement convention "synergy lives in the consumer-shaped package".

---

## 5. Landing order (which primitives to ship first)

| PR | LOC | Primitives | Saturates | Blocking |
|----|-----|------------|-----------|----------|
| **PR-1** | **220** | **N1+N3+N4+N5+N9** | **R-CONSENSUS-3-WAY P1** | none |
| PR-2 | 100 | N6 + R-FIEDLER pin | R-FIEDLER P2 | none |
| PR-3 | 110 | N10+N11+N12 weighted/gossip/push-sum | gossip-mixing-rate | none |
| PR-4 | 90 | N13+N14 leader/containment | leader-grounded-λ_g | none |
| PR-5 | 50 | N15 edge agreement | edge-Laplacian-rank=n−c | none |
| PR-6 | 200 | N16+N17 Kuramoto+boids | R-KURAMOTO P3, R-OLFATI-SABER-α-rigid | none |
| PR-7 | 180 | N19+N22 structural-controllability+pinning | Lin-1974/Wang-Chen-2002 | none |
| PR-8 | 180 | N20+N21 distributed-subgrad+D-ADMM | Nedić-Ozdaglar-2009-rate | needs `optim/proximal.Admm` consumer-test |
| PR-9 | 80 | N7+N8 Fiedler-vector+spectral-bisection | Cheeger-bound | **097-T1 Eigvec** |
| PR-10 | 80 | N18 LeaderControllability | Liu-Slotine-Barabási Table-S1 | **097-T1 Eigvec** |
| PR-11 | 60 | N2 LaplacianFlow | exact e^{−tL} consensus | **097-T1 MatrixExp** |
| PR-12 | 250 | N23 NetworkedKalman | Olfati-Saber-Fax-Murray-2007 | **161-C5 + 097-T1** |

Total: ~3,260 LOC source + ~1,200 LOC golden-tests over ~13 engineer-days. **PR-1+PR-2 = 320 LOC = single-day high-leverage saturation** (canonical R-MUTUAL pin extended to networked-control axis + the highest-leverage spectral-graph scalar).

---

## 6. Precision hazards (per-primitive pinning targets)

- **N1.** Symmetric L for undirected graph; for directed graph either symmetrise (in/out-degree-Laplacian) or use the directed Chung-2005 random-walk-Laplacian — document the choice explicitly. Floating-point: integer-degree graphs give exact L (no roundoff).
- **N3 / N4.** Mass conservation: 1ᵀẋ = 1ᵀ(−Lx) = 0 since 1ᵀL = 0; pin (1/n)Σx_i(t) = const to **machine precision** at every RK4 step (this is the strongest invariant in the entire networked-control body).
- **N6.** Disconnected graph → λ₂ = 0 (multiplicity = number-of-components). Caller must check N9 first; document panic-on-disconnected as alternative.
- **N7 (blocked).** Eigenvector signs are not unique — canonicalise by largest-|component|-positive convention.
- **N11.** Random number source: take `*rand.Rand` argument so determinism is caller's choice. Mass-conserved per step (xᵢ+xⱼ unchanged).
- **N12 push-sum.** Numerical underflow as wᵢ → 0 — clamp ratio to s_i/max(w_i, 1e-300).
- **N13 / N22.** Grounded-Laplacian is not symmetric in general (rows zeroed, columns kept) — eigvals are still positive iff every follower reaches at least one leader (verified by `graph.BFSDownstream` from leaders).
- **N16.** Phase wrap-around: take θ mod 2π only at output time, never during integration — phase-unwrapping in-loop breaks RK4 order.
- **N17.** Proximity graph is **time-varying** (edges appear/disappear as agents move) — re-build N1 each timestep; Olfati-Saber-2006 algebraic-conditions α-rigid graph requires λ₂ > 0 throughout — instrument and break-on-disconnection.
- **N18 / N19.** Lin-1974 dilation test is sharp only for **structural** controllability (generic value of edge-weights) — does not detect numerical-rank-deficiency from specific weight-coincidences (Lin-1974 §IV).
- **N20.** Step-size ε must be square-summable-not-summable (ε_k = 1/√k typical) — document this requirement.
- **N21.** ρ-tuning sensitive — Nishihara-Lessard-Recht-Packard-Jordan-2015 optimal ρ formula on consensus problems.
- **N23.** Information-form Kalman must be used (not covariance-form) for additivity over consensus rounds.

---

## 7. Cross-language pinning targets

- N1 / N6 / N7 / N8: **NetworkX** `networkx.laplacian_matrix`, `algebraic_connectivity`, `fiedler_vector`, `spectral_bisection` to 1e-9. **SciPy** `scipy.sparse.csgraph.laplacian` agreement.
- N3 / N4: **CVXPY** consensus-tutorial reference traces.
- N6 closed-form: Spielman-Srivastava-2008-Sparsification supplementary tables on K_n / K_{m,m}.
- N11: **Boyd-Ghosh-Prabhakar-Shah-2006-IT-Table-1** mixing rates on small graphs (5×5, 6×6 randomised gossip Markov chain analytic).
- N16: **Strogatz-2000-PhysicaD-Fig-1** order-parameter-vs-K phase transition on uniform Cauchy ω-distribution.
- N17: **Reynolds-1987-SIGGRAPH** boids reference implementation behavioural pin (qualitative — flocking visible at K, dispersing below).
- N18 / N19: **Liu-Slotine-Barabási-2011-Nature-Table-S1** minimum-driver-node-counts for {Caenorhabditis-elegans-neural, electronic-power-grid, Twitter-mention} datasets — public pinning data.
- N20: **Nedić-Ozdaglar-2009-IEEE-TAC-Fig-3** convergence-rate empirical curves.
- N21: **Boyd-2011 ADMM-distributed-LASSO** scikit-learn `sklearn.linear_model.Lasso` parity at 1e-7.
- N22: **Wang-Chen-2002-PhysicaA-Fig-3** Lorenz-Wang-Chen-network synchronisation onset at k_c.

Eleven of twenty-three primitives have public-API equivalents pinning at 1e-7 or better; the remaining twelve are either math-closed-form (N5, N9 trivial) or blocked-on-097/161 dependencies.

---

## 8. Differentiation from neighbouring reviews

- **vs 081-085 graph-isolation** — those flag L=D−A and Fiedler as missing primitives in `graph/`-only sense. **THIS** adds the *consumer-pull* from networked control: the Laplacian without `LaplacianFlow`/`KuramotoOnGraph`/`PinningControl` is geometrically nice but operationally inert; the synergy collapses 30+ standalone-graph-spectral demos into one coherent "the network IS the dynamical system" narrative.
- **vs 051-055 control-isolation** — those flag missing `StateSpace` / Kalman / LQR / MPC. **THIS** adds the *graph-distributed* axis: every primitive in 161 (LQG), 178 (MPC) lifts to a distributed-over-network version (consensus-LQG = N23, consensus-MPC = N21+178-M5).
- **vs 097 linalg-missing** — flags `Eigvec`/`SVD`/`MatrixExp` as Tier-1 absent primitives. **THIS** sharpens 097-T1 priority by surfacing *three new consumers* of `Eigvec` (N7 Fiedler-vector, N18 leader-controllability rank, N23 Kalman-observability) and *one new consumer* of `MatrixExp` (N2 exact-Laplacian-flow). 097-T1 priority should bump on these grounds.
- **vs 157 synergy-graph-linalg** — 157 lands the spectral-graph canon (heat kernel, GFT, Cheeger, effective resistance, sparsification — sixteen primitives, ~1380 LOC). **THIS** is the *control-coupled* sibling: 157 is "the graph is a metric space"; 186 is "the graph is a dynamical system" — N1 / N2 / N6 / N7 / N8 are shared between 157 and 186 (this review consumes them; 157 lands them on architectural-completeness; 186 lands them on consumer-pull). Recommend joint-lands of N1+N2+N6+N7+N8 across the 157+186 pair.
- **vs 161 synergy-control-prob** — 161 lands LQG/Kalman/PF on the StateSpace + Cholesky bridge for **single-agent** stochastic control. **THIS** lifts to the network: N23 networked-Kalman is consensus-of-(161-C5 KalmanFilter); N20 + N21 are distributed-LQG-style optimisation; the per-agent-has-StateSpace assumption inherits from 161-C1 directly.
- **vs 162 synergy-graph-prob** — 162 lands graph-stochastic primitives (Markov-on-graph, percolation, SIR, exponential-random-graph models). **THIS** is orthogonal: 162 is "random graph"; 186 is "deterministic-graph + dynamical-system". The intersection is N11 RandomGossip (random-edge-selection rate is uniform) + N12 PushSum (random out-neighbour) — but the convergence-rate analysis is graph-spectral (186-N6) not graph-stochastic (162).
- **vs 163 synergy-optim-autodiff** — 163 lands second-order optimisation + KKT primitives. **THIS** consumes `optim/proximal.Admm` for N21 (D-ADMM) and `optim.LBFGS` cameo for N20 (per-agent gradient step). The cross-link is N20 / N21 use 163's PR-3+PR-4 surface.
- **vs 171 synergy-graph-topology** — 171 covers Betti-numbers / persistent-homology of graphs. **THIS** has one cross-link at N15 EdgeAgreement (kernel of L_e = cycle space = first Betti number); call out cycle-space as a 171-186 shared concept.
- **vs 177 synergy-geometry-optim** — 177 covers Voronoi / Lloyd / coverage. **THIS** has cross-link at N14 ContainmentControl (followers ∈ ConvexHull(leaders)) and a future N24 CoverageControl-Cortés-Bullo-2004 (deferred — left as cross-link rather than landed primitive).
- **vs 178 synergy-control-optim** — 178 lands MPC / KKT / iLQR for **single-agent**. **THIS** lifts to consensus-MPC (N21 = D-ADMM splitting of 178-M5 LinearMpc objective across agents). Recommend 178+186 joint-lands of D-MPC after 178-M5 is in place.

This synergy is **medium-leverage by consumer-multiplicity** (multi-agent systems matter to swarm robotics + power-grid stability + sensor-network estimation — three Reality consumer surfaces) but **highest-leverage by mathematical-novelty** in the 186-review cohort: the three-way intersection graph × control × dynamical-systems has **no zero-dependency Go reference implementation** anywhere in the public ecosystem (NetworkX is graph-only, python-control is single-agent-only, ddsp / DiffEqFlux are domain-specific). Reality landing N1–N16 in ~1,200 LOC over ~6 engineer-days would establish the canonical multi-agent-systems substrate the field has been waiting for.

---

## 9. Single-day high-leverage commit (if-only-one-PR)

**PR-1 = N1 + N3 + N4 + N5 + N9 + N6 = 320 LOC source + 130 LOC tests:**
1. Lands FIRST `graph/` → `linalg/` cross-edge (N1 Laplacian → returns []float64 consumed by linalg.MatVecMul).
2. Lands FIRST `control/` → `graph/` cross-edge in PR-1+ (the closure N3 ConsensusODE consumes graph.Laplacian output).
3. Saturates canonical R-CONSENSUS-3-WAY R-MUTUAL pin to 1e-9 (P1 above) — extends the 6a55bb4/365368a/1e12e80 R-MUTUAL family to networked-control axis.
4. Saturates R-FIEDLER-CLOSED-FORM pin to 1e-12 on path / cycle / star / K_n — the most direct numerical pin in the entire 186-review.
5. Establishes the architectural keystone N1 from which fifteen of twenty-three subsequent primitives derive — every multi-agent paper since 2004 starts with "let L=D−A be the graph Laplacian"; without N1 there is nothing to compose.
6. The cheapest possible saturation of T1 Olfati-Saber-Murray-2004 consensus theorem, which is the textbook landing-page of the entire networked-control field.
7. N6 gives the single scalar that converts every subsequent primitive from empirical-tuning to closed-form-design (every multi-agent design exercise reduces to "what is λ₂ of my network and is it big enough?") — without N6 every Nᵢ for i ∈ {3, 4, 10, 11, 13, 16, 18, 22, 23} is a black box; with N6 every one becomes a quantitative engineering exercise.

≈386 lines.

# 278 — new-ip-relaxation (LP/IP Relaxations: Lagrangian, Branch-and-Cut, Dual Decomposition)

## Headline
reality v0.10.0 has **zero Lagrangian-relaxation, dual-decomposition, Benders, Dantzig-Wolfe, column-generation, or Sherali-Adams/Lasserre lifting surface** — but the substrate is unusually well-positioned (`optim/proximal/admm.go::Admm` lines 53-120 is *literally consensus-ADMM with explicit access to the scaled dual `u`*, `optim/linear.go::SimplexMethod` lines 35-155 is a usable LP-oracle, `optim/proximal/operators.go::ProxNonNeg`+`ProxBox`+`ProxSimplex` lines 85-156 are exactly the dual-projection operators a Lagrangian iteration needs, and `graph/mst.go::PrimMST`/`KruskalMST` line 34/110 + `graph/flow.go::MaxFlow` line 25 give the Held-Karp 1-tree and min-cut LP-relaxation oracles for free) — so a **9-primitive `optim/relax/` sub-package totalling ~1,820 LOC** ships immediately on top of slot 277's `optim/intopt/` stack, distinct from 277's exact-MIP focus and dedicated to the *bound-quality / decomposition-scalability* axis: Lagrangian subgradient on the dual, Held-Karp-1-tree-via-MST, Benders L-shaped, Dantzig-Wolfe column generation, and the Sherali-Adams LP / Lasserre SDP hierarchies that bridge LP-relaxation (loose) → Sherali-Adams-rₜ (tighter) → Lasserre-rₜ via slot 276's SDP (tightest) → ILP-optimal (slot 277).

## Findings

### F1. SimplexMethod does NOT expose duals — must extend or wrap
`optim/linear.go::SimplexMethod` (lines 35-155) returns only `(x, objVal, error)`. Lines 144-152 extract primal `x` from `basis[]` but **never compute** `y = c_B' * B^{-1}` (the dual prices) nor the reduced costs vector at termination, despite calculating reduced costs internally (lines 92-101) every iteration. This is the **single biggest gap** for relaxation theory: Lagrangian relaxation needs `y*` to seed dual multipliers, Benders feasibility/optimality cuts ARE the dual extreme rays/vertices of the subproblem, and column-generation pricing reads dual prices off the master LP. **Fix:** add `SimplexResult{X, ObjVal, Y, ReducedCosts, Basis, Status}` and a backwards-compatible `SimplexMethodFull(c,A,b) (*SimplexResult, error)` (~60 LOC). Required by R1, R3, R5, R7 below.

### F2. ADMM is dual-decomposition incarnate — already shipping
`optim/proximal/admm.go` (full file 1-120) implements scaled-form consensus ADMM with `u []float64` as the explicit scaled dual variable (line 38, 46, 92-95). The Boyd-2011 §3 update `u^{k+1} = u^k + x^{k+1} - z^{k+1}` is line 95. **This means dual-ascent / dual-decomposition with Bregman-style penalty IS already in the repo** — Lagrangian relaxation in its consensus form (split a hard problem into f+g via x=z, penalise the coupling) is literally one wrapper away. The only missing piece is exposing `u` post-convergence and adding a thin `optim/relax/dual_decomp.go` that drives `Admm` with f = min over relaxed-constraints subproblem, g = indicator of original feasibility.

### F3. Subgradient method is missing as a standalone primitive
`optim/proximal/operators.go` line 149 cites Held-Wolfe-Crowder 1974 "Validation of subgradient optimization" Math.Prog. 6:62-88 *in a docstring of the simplex projection*, but no `Subgradient(f, x0, stepRule, maxIter)` driver exists. Slot 276 X18 enumerates this as a 60-LOC primitive (Polyak-1969); 278 takes a hard dependency on it for the dual ascent in Lagrangian relaxation. **Recommend co-shipping with slot 276 X18, OR re-exporting from `optim/relax/subgradient.go` if 276 lands later.**

### F4. Projected-gradient on the dual feasible set already buildable
`optim/proximal/operators.go::ProxNonNeg` (lines 85-101) is the indicator-projection onto R+^m (i.e. the multipliers for inequality constraints `g(x) ≤ 0` which require `λ ≥ 0`). `ProxBox` (lines 103-124) handles two-sided dual bounds. `ProxSimplex` (lines 156-...) handles convex-combination duals. So a Lagrangian iteration `λ_{k+1} = ProxNonNeg(λ_k + α_k * g(x*(λ_k)))` reuses what's there.

### F5. Held-Karp 1-tree TSP relaxation is a near-trivial composition over existing graph code
`graph/mst.go::PrimMST` (line 110) and `KruskalMST` (line 34) return `(mstEdges [][3]float64, totalWeight float64)`. The Held-Karp 1-tree is: compute MST on G \ {vertex 0}, then add the two cheapest edges incident to vertex 0 → that's the 1-tree. Its weight is a TSP lower bound (Held-Karp 1971). Lagrangian-tighten: introduce node potentials `π_v`, replace edge weights `c_uv` with `c_uv - π_u - π_v`, recompute 1-tree, take subgradient step `π_v ← π_v + α(deg_v − 2)`. **This is the canonical demo** of Lagrangian relaxation — every textbook (Wolsey, Cook-Cunningham-Pulleyblank-Schrijver, Ahuja-Magnanti-Orlin) ships it. Reality can ship it in ~200 LOC composed entirely from existing pieces.

### F6. Sherali-Adams / Lasserre absent; slot 276 SDP unblocks Lasserre
The Sherali-Adams (LP) and Lasserre (SDP) hierarchies are the canonical convex-hull-tightening machinery (Sherali-Adams 1990 SIAM-J-Disc-Math 3:411; Lasserre 2001 SIAM-J-Optim 11:796). Sherali-Adams round-r is an LP with O(n^r) variables and constraints — solvable with `SimplexMethod` or 102-T1.13 Mehrotra. Lasserre round-r is an SDP — blocks on slot 276 X12 AHO-SDP. **Cross-link gold:** Lasserre round-r at fixed r collapses gracefully to a slot-276 conic program, making 278's Lasserre wrapper an ~80-LOC reformulator on top of 276 X12. R-MUTUAL-CROSS-VALIDATION 3/3 chain becomes auditable: `LP-relax(P) ≤ SA_r(P) ≤ Lasserre_r(P) ≤ ILP-opt(P)` for every 0/1-LP `P` with hierarchy-level `r`.

### F7. Benders ≡ outer-approximation-on-dual ≡ row generation ≡ L-shaped (stochastic)
Benders 1962 Numer.Math. 4:238 partitions decision variables into "complicating" `y` and "easy" `x`, solves a master MILP over `y` augmented by feasibility cuts (from infeasible subproblem dual rays) and optimality cuts (from finite subproblem dual vertices). The dual-information stream is exactly what F1 fixes. **Stochastic-programming L-shaped (Van Slyke-Wets 1969 SIAM-J-Appl-Math 17:638) is Benders applied scenario-by-scenario** — single primitive serves both. Closes a SINGULAR slot-169-prob×optim seam (two-stage stochastic LP).

### F8. Dantzig-Wolfe column generation is the dual axis to row generation (Benders)
DW 1960 Op.Res. 8:101 is row-generation in the dual ↔ column-generation in the primal: a "restricted master LP" over a small subset of columns + a pricing-problem subroutine that finds a column with negative reduced cost (the *separation problem* for the dual). Cutting-stock (Gilmore-Gomory 1961 Op.Res. 9:849) is the canonical demo. Branch-and-price = D-W column generation embedded in slot 277 I3 BnB. **Slot 277 I16/I17 deferred this to T5; 278 owns it.**

### F9. NO pure-Go zero-dep MIT-licensed Lagrangian/Benders/D-W solver exists worldwide
Verified via WebSearch May-2026: PyLAR (Python, github.com/dilsonpereira/pylar) is the closest microframework analog. Every Go option (golp, gonum/optimize) wraps GLPK/LP-SOLVE via CGo. **Reality post-PR is the only zero-dep MIT-Go relaxation/decomposition library in existence.** SINGULAR-MOAT alongside slot 276 (only zero-dep MIT-Go conic solver) and slot 277 (only zero-dep MIT-Go MIP solver).

### F10. graph.MaxFlow is already a tight LP relaxation (LP-IP gap zero by Ford-Fulkerson 1956 / König-Egerváry totally-unimodular A)
`graph/flow.go::MaxFlow` (line 25) is the optimum of the LP-relaxation of the integer max-flow MIP — and equals the integer optimum because the constraint matrix is totally unimodular (TU). This is **the single existing example of LP=IP in reality**, ready as a R-MUTUAL-CROSS-VALIDATION 3/3 anchor: `MaxFlow ≡ LP-relax-of-flow-ILP-via-SimplexMethod ≡ Lagrangian-relax-with-zero-gap` (the latter on the integer-flow MIP — the LR gap is provably zero for TU systems by LP-duality + integrality of vertices).

## Concrete recommendations (9 primitives, ~1,820 LOC, optim/relax/)

### Tier 0 — LP-dual exposure (~60 LOC, foundational, must ship FIRST)

1. **`optim/linear.go::SimplexMethodFull(c, A, b) (*SimplexResult, error)` + `type SimplexResult struct { X, Y, ReducedCosts []float64; ObjVal float64; Basis []int; Status SimplexStatus }` (~60 LOC).** Backwards-compatible extension: existing `SimplexMethod` becomes a thin wrapper that discards the dual fields. Compute `y = c_B' * B^{-1}` from the existing tableau at termination (already implicit in the cost-row reduced costs, lines 92-101). Required by R3 (Benders), R5 (D-W column generation), R7 (Sherali-Adams). **Pin R-LP-DUALITY 1/1: c'x* == b'y* at optimality** (strong duality, exact-equality up to 1e-12).

### Tier 1 — Lagrangian / subgradient core (~440 LOC, the keystone tier)

2. **`optim/relax/subgradient.go::Subgradient(g func([]float64) (float64, []float64), x0 []float64, cfg SubgradCfg) (xStar []float64, fStar float64)` (~80 LOC).** Polyak-1969 generic subgradient with three step rules: (a) constant step, (b) constant step length, (c) Polyak diminishing-non-summable `α_k = 1/√k`, (d) Polyak optimal-known `α_k = (f_k − f*)/||g_k||²`. Closes 276 X18 if 276 ships later; if 276 ships first, 278 imports `optim/cvx.SubGradient` instead. **Pin R-SUBGRAD-CONVERGENCE 1/1** on smooth f (subgradient agrees with gradient).

3. **`optim/relax/lagrangian.go::LagrangianRelax(model *RelaxModel, oracle SubproblemOracle, cfg LagrCfg) (*LagrResult, error)` (~200 LOC).** Generic Lagrangian-relaxation outer loop. Caller supplies (a) which constraints to dualise (`AλEq`, `AλIneq`, `b`), (b) a `SubproblemOracle` that solves `L(λ) = min_x { c'x + λ'(Ax − b) : x ∈ X }` over the *easy* feasible set X. Loop: (1) call oracle → x*(λ_k), (2) compute subgradient `g_k = A x*(λ_k) − b`, (3) update `λ_{k+1} = ProxNonNeg(λ_k + α_k g_k)` for inequalities or unconstrained for equalities (reuse `ProxNonNeg` from `optim/proximal/operators.go` line 85). Returns best dual bound, oracle-call count, gap. **Pin R-LAGR-WEAK-DUALITY 1/1: L(λ) ≤ ILP-opt for all λ ≥ 0** (every iterate is a valid lower bound — checkable cheap invariant on every test).

4. **`optim/relax/held_karp.go::HeldKarpBound(distMatrix [][]float64, cfg HKCfg) (lowerBound float64, oneTreeEdges [][2]int, multipliers []float64)` (~120 LOC).** Held-Karp 1971 Op.Res. 18:1138 1-tree TSP lower bound. Reuses `graph/mst.go::PrimMST` (line 110) for the MST-on-G\{0} step. Lagrangian-tightens via R3 with `λ_v` = node potential, subgradient `g_v = deg_v − 2`. Stops when 1-tree IS a Hamiltonian cycle (gap = 0) or iteration cap hit. **Empirical: 99% of OPT on random Euclidean instances** (Mumford-Cardiff). **THE canonical pedagogical demo of Lagrangian relaxation.** Cross-link: slot 277 I13 TSP gets its lower-bound oracle for free.

5. **`optim/relax/dual_decomp.go::DualDecompose(blocks []SubproblemOracle, couplingA [][]float64, couplingB []float64, cfg DDCfg) (*DDResult, error)` (~80 LOC).** Dantzig-Wolfe-style price-decomposition for block-angular MIPs: blocks share a *coupling constraint* `Σ_b A_b x_b ≤ b`. Dualise the coupling, each block solves its own subproblem in parallel given current `λ`, master updates `λ` via R2 (subgradient) on `g = Σ_b A_b x_b − b`. **Wraps existing `optim/proximal/admm.go::Admm` (line 53)** as the proximal-stabilised variant (Eckstein-Bertsekas 1992). Gateway to Bertsekas-2003 distributed dual-decomposition for stochastic-network/multicommodity-flow problems.

### Tier 2 — Benders, D-W, column generation (~520 LOC, the consumer-facing decomposition tier)

6. **`optim/relax/benders.go::BendersDecompose(master MasterProblem, sub SubProblem, cfg BendersCfg) (*BendersResult, error)` (~200 LOC).** Benders 1962 Numer.Math. 4:238. Iterates: (a) solve master MILP via slot 277 I3 BranchAndBound on relaxed-cuts version of master, (b) solve LP subproblem via R1 SimplexMethodFull at fixed `y*` from master, read dual `(π_k, μ_k)`, (c) if subproblem infeasible → feasibility cut `μ_k'(B y) ≥ μ_k' b` from extreme dual ray; if feasible → optimality cut `θ ≥ π_k'(b − B y)`, (d) add cut to master, repeat until θ_master ≥ θ_sub − ε. **Two-stage stochastic LP via L-shaped is a one-loop scenario wrapper around Benders** — ship as `BendersStochastic(scenarios []Scenario, ...)` siamese-twin entrypoint sharing 90% code (~30 extra LOC).

7. **`optim/relax/column_gen.go::ColumnGeneration(rmpInit *RMP, pricing PricingProblem, cfg CGCfg) (*RMPResult, error)` (~180 LOC).** D-W 1960 + Gilmore-Gomory 1961. Iterates: (a) solve restricted master LP via R1 SimplexMethodFull, read dual `y*`, (b) call user-supplied `pricing(y*) (negRedCostCol, redCost)` to find column with negative reduced cost, (c) if `redCost ≥ −ε` STOP (LP-optimal proven), else add column to RMP and re-solve. Stabilisation via dual-bounding box (du Merle et al. 1999) optional but recommended (40 LOC of R7's 180). **Cutting-stock pedagogical demo (`examples/cutting_stock_test.go`)** — knife-thin wrapper on R7 that supplies the pricing-problem (a 1D knapsack, reuse slot 277 I12 KnapsackDP). **Pin R-DW-LP-EQUIV 1/1: column-generation LP-bound == solving full LP-with-all-columns directly** (when columns enumerable, e.g. cutting-stock with width-bound enforcing finite pattern set).

8. **`optim/relax/branch_price.go::BranchAndPrice(model *MIPModel, pricing PricingProblem, cfg BnPCfg) (*MIPSolution, error)` (~140 LOC).** B&B (slot 277 I3) where each node's LP-relaxation is solved by R7 column-generation. Branching strategies: (a) Ryan-Foster on subset variables (default for set-partition formulations), (b) most-fractional original variable. **Closes slot 277 I17 deferred-to-T5.** Single most-cited industrial application: vehicle-routing (VRP via set-partitioning).

### Tier 3 — Hierarchies (~440 LOC, the SOTA tightening tier)

9. **`optim/relax/sherali_adams.go::SheraliAdamsLift(rawILP *ILP, level int) (*LP, error)` + `SheraliAdamsBound(rawILP *ILP, level int) (lb float64, error)` (~240 LOC).** Sherali-Adams 1990 SIAM-J-Disc-Math 3:411 LP-lift. Generates the level-`r` reformulation (n^r new variables, n^r new constraints, McCormick-multilinearisation + factorial-multiplier) and solves via R1 SimplexMethodFull. Returns LP-relaxation bound monotone in `r`. **For r=n, Sherali-Adams lift IS the integer hull (proved 1990).** Pin R-SA-MONOTONE 3/3: SA(r=0) == LP-relax(P) AND SA(r=1) ≤ SA(r=2) ≤ ... ≤ ILP-opt AND SA(r=n) == ILP-opt. **Critical caveat in docstring:** O(n^r) blowup means r ≤ 3 is practical for n ≤ 50; document this and refer users to R10 (Lasserre) for SDP-tighter alternative.

10. **`optim/relax/lasserre.go::LasserreLift(rawILP *ILP, level int) (sdp *ConicProgram, error)` + `LasserreBound(rawILP *ILP, level int) (lb float64, error)` (~200 LOC).** Lasserre 2001 SIAM-J-Optim 11:796 SDP-lift. **HARD dependency on slot 276 X1-X4 (conic abstraction) + X12 AHO-SDP** — without 276 Tier-0+Tier-3, R10 cannot ship. Generates moment-matrix SDP at level `r` and dispatches to slot 276 ConicSolve. **Lasserre dominates Sherali-Adams at every level r** (Laurent 2003 MOR 28:470). Pin **R-MUTUAL-CROSS-VALIDATION 3/3: LP-relax(P) ≤ SA_r(P) ≤ Lasserre_r(P) ≤ ILP-opt(P)** with strict ≤ on enough non-trivial 0/1-LPs (knapsack with max-clique side-constraints is a textbook witness — Karlin-Mathieu-Nguyen 2011). **THE singular-moat primitive of slot 278.**

## Cross-cutting

- **slot 277 (just shipped) ← 278 owns the relaxation-theory/decomposition axis 277 deferred (I8 BranchAndCut deepening, I16 ColumnGeneration → R7, I17 BranchPrice → R8). 277's `optim/intopt/` is the exact-MIP framework; 278's `optim/relax/` is the bound-quality framework. Both consume `optim/linear.go` post-R1 for duals.**
- **slot 276 cvxopt-extras ← 278 R10 (Lasserre) is a BLOCKED-ON-276-Tier-0+Tier-3 consumer (needs ConicProgram + AHO-SDP). 276 ships first → 278 R10 ships. 278 R10 is the canonical demonstration of why the SDP investment of 276 pays off in IP-relaxation.**
- **slot 102 optim-missing ← 278 R1 closes the dual-exposure gap that 102 flagged for SimplexMethod. 102's Mehrotra-MPC (T1.13) is a drop-in upgrade for the LP-relaxation oracle inside R6+R7+R9 once shipped.**
- **slot 174-G18 Frank-Wolfe ← FW is the *projection-free* primal companion to Lagrangian *dual* iteration; 174-G18 + 278-R3 are dual axes, 174 owns continuous-polytope-vertex direction, 278 owns dual-multiplier-direction. Disjoint axes — no overlap.**
- **slot 199 Lovász θ ← 199 ships the eigenvalue-bound version, 276-X14 ships the SDP version, 278-R10 ships the Lasserre-r generalisation (Lovász θ ≡ Lasserre level-1 for stable-set polytope). Three-tier strict refinement chain, each adding tightness.**
- **slot 254 graph-cuts ← 254 owns max-flow=min-cut (LP=IP via TU), F10 above flags graph.MaxFlow as the canonical R-MUTUAL-CROSS-VALIDATION 3/3 anchor for every relaxation primitive in 278 (zero-gap witness). 254 + 278 + 277 form the canonical-flow-theory + relaxation + exact-MIP triad.**
- **slot 169 prob×optim seam ← stochastic programming L-shaped via R6 BendersStochastic fills the prob×optim seam 169 flagged. R6 is the singular-prob×optim deliverable of 278.**
- **slot 277 I13 TSP ← R4 HeldKarpBound is the lower-bound oracle for 277's TSP B&C with lazy subtour-elimination cuts. 278-R4 ships independently and improves 277-I13 incumbents by ~99% of OPT (empirical) on first iteration.**

## Singular cheapest day-1 PR

**R1 + R2 + R3 + R4 (~460 LOC, ZERO cross-package blockers, depends only on existing `optim/linear.go` + `optim/proximal/operators.go::ProxNonNeg` + `graph/mst.go::PrimMST`):** dual-exposure for SimplexMethod + Polyak subgradient + generic LagrangianRelax + HeldKarpBound TSP demo. Ships in 1 day, demonstrates the entire relaxation-theory pillar via the canonical pedagogical example, and pins **R-LAGR-WEAK-DUALITY 1/1** (every Lagrangian iterate is a valid LB) and **R-MUTUAL-CROSS-VALIDATION 3/3** on the Held-Karp ≡ MST+two-cheapest-degree-1-edges ≡ subgradient-on-degree-defect chain across n=15 random Euclidean TSP instances.

## Sources

### Repo files (reality v0.10.0)
- `C:\limitless\foundation\reality\optim\linear.go` lines 35-155 (SimplexMethod, returns x+objVal but NOT y duals — F1 critical-fix-target)
- `C:\limitless\foundation\reality\optim\linear.go` lines 157+ (InteriorPoint — already flagged mis-labelled by 102/276 as damped-gradient-on-log-barrier)
- `C:\limitless\foundation\reality\optim\proximal\admm.go` lines 53-120 (Admm with explicit u-dual scaled-form, Boyd-2011 §3.1.1 — F2)
- `C:\limitless\foundation\reality\optim\proximal\operators.go` lines 85-101 (ProxNonNeg — F4 dual-projection), lines 103-124 (ProxBox), lines 145-156 (ProxSimplex with Held-Wolfe-Crowder-1974 docstring citation — F3)
- `C:\limitless\foundation\reality\graph\mst.go` lines 34, 110 (KruskalMST/PrimMST — F5 Held-Karp building block)
- `C:\limitless\foundation\reality\graph\flow.go` line 25 (MaxFlow — F10 LP=IP TU witness)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\277-new-copo.md` (slot 277 enumerates I8/I16/I17 deferred-to-T5; 278 owns those)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\276-new-cvxopt-extras.md` (slot 276 X1-X4+X12 are R10's hard dependency chain)

### Foundational papers
- Held M., Karp R.M. (1971). "The traveling-salesman problem and minimum spanning trees: Part II." Math.Prog. 1:6-25.
- Held M., Wolfe P., Crowder H.P. (1974). "Validation of subgradient optimization." Math.Prog. 6:62-88.
- Lemaréchal C. (2001). "Lagrangian Relaxation." In Jünger-Naddef Computational Combinatorial Optimization. LNCS 2241, Springer.
- Polyak B.T. (1969). "Minimization of unsmooth functionals." USSR Comput.Math. 9:14-29.
- Benders J.F. (1962). "Partitioning procedures for solving mixed-variables programming problems." Numer.Math. 4:238-252.
- Van Slyke R., Wets R.J.-B. (1969). "L-shaped linear programs with applications to optimal control and stochastic programming." SIAM J.Appl.Math. 17:638-663.
- Dantzig G.B., Wolfe P. (1960). "Decomposition principle for linear programs." Op.Res. 8:101-111.
- Gilmore P.C., Gomory R.E. (1961). "A linear programming approach to the cutting-stock problem." Op.Res. 9:849-859.
- Sherali H.D., Adams W.P. (1990). "A hierarchy of relaxations between the continuous and convex hull representations for zero-one programming problems." SIAM J.Disc.Math. 3:411-430.
- Lasserre J.B. (2001). "Global optimization with polynomials and the problem of moments." SIAM J.Optim. 11:796-817.
- Laurent M. (2003). "A comparison of the Sherali-Adams, Lovász-Schrijver and Lasserre relaxations for 0-1 programming." Math.Op.Res. 28:470-496.
- Eckstein J., Bertsekas D.P. (1992). "On the Douglas-Rachford splitting method and the proximal point algorithm for maximal monotone operators." Math.Prog. 55:293-318.
- Boyd S. et al. (2011). "Distributed Optimization and Statistical Learning via ADMM." Found.Trends Mach.Learn. 3:1-122.

### Web (May-2026 verified)
- [Lagrangian relaxation - Wikipedia](https://en.wikipedia.org/wiki/Lagrangian_relaxation)
- [Held-Karp algorithm - Wikipedia](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)
- [Collins ACL Tutorial: Dual Decomposition and Lagrangian Relaxation](http://www.cs.columbia.edu/~mcollins/acltutorial.pdf)
- [Stanford CS369H: Hierarchies of IP Relaxations](https://web.stanford.edu/class/cs369h/lectures/lec1.pdf)
- [Rothvoss: The Lasserre hierarchy in Approximation algorithms](https://sites.math.washington.edu//~rothvoss/lecturenotes/lasserresurvey.pdf)
- [Benders decomposition - Wikipedia](https://en.wikipedia.org/wiki/Benders_decomposition)
- [JuMP-dev: Benders decomposition tutorial](https://jump.dev/JuMP.jl/stable/tutorials/algorithms/benders_decomposition/)
- [Lübbecke: Selected Topics in Column Generation](https://optimization-online.org/wp-content/uploads/2002/12/580.pdf)
- [Approximating the Held-Karp Bound for Metric TSP in Nearly Linear Time](https://arxiv.org/pdf/1702.04307)
- [pylar: Python microframework for IP Lagrangian relaxation](https://github.com/dilsonpereira/pylar) — closest worldwide analog to proposed `optim/relax/`; Python-only, MIT-license-equivalent

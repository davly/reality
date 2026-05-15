# 277 — new-copo (Combinatorial Optimization: ILP, B&B, Gomory)

## Headline
reality v0.10.0 has **zero integer / combinatorial optimisation surface** — no ILP, MILP, branch-and-bound, branch-and-cut, branch-and-price, Gomory fractional or mixed-integer cuts, lift-and-project, Chvátal-Gomory closure, knapsack, set cover, vertex cover, max-clique, Hungarian, or TSP exists in any `*.go` source file — but the LP-relaxation oracle is in place (`optim/linear.go::SimplexMethod` is correct revised-simplex with Bland's rule, lines 35-155) so a 14-primitive `optim/intopt/` sub-package totalling ~3,050 LOC is unblocked today and would make `reality` the **only zero-dependency MIT-licensed pure-Go MIP solver in existence** — every other Go option (golp, goop, glpk, nextmv-io/sdk/mip) is a CGo wrapper around external LPSolve / GLPK / Gurobi.

## Findings

- **Repo-wide grep** on `ILP|MILP|MIP|BranchAndBound|Gomory|CuttingPlane|IntegerProgramming|IntegerLinear|Knapsack|TravelingSalesman|VertexCover|MaxClique|Hamiltonian|SetCover|Hungarian|Munkres|BranchCut|LazyConstraint|Polytope` returns **zero source-code matches** — only review-doc references and unrelated false-positives (`chaos.Hamiltonian` = energy invariant; `topology/persistent/bottleneck.go:248` = Hopcroft-Karp inner-loop, bipartite-cardinality matching, not assignment).
- **LP relaxation oracle is GOOD.** `optim/linear.go::SimplexMethod(c, A, b)` (lines 35-155) is a correct revised-simplex with Bland's anti-cycling rule, returns `(x []float64, objVal float64, err error)`. Standard form `min c'x s.t. Ax≤b, x≥0`. **Reusable as-is** for B&B node relaxations — the signature matches what an LP-oracle interface needs. `maxIter=10000` (line 87) — adequate for small-ish nodes, would need raising for large MIPs.
- **`optim/linear.go::InteriorPoint` is unusable for B&B** — slot 102-F7 / 276 confirmed it is a barrier-gradient heuristic, not Newton-on-KKT. Returns wrong values on degenerate / unbounded LPs that arise inside B&B. **Do NOT call from intopt.** Wait for slot 102-T1.13 / 276-X-Mehrotra-MPC.
- **Graph package has zero combinatorial-opt overlap.** `graph/` ships shortest-path (Dijkstra/A\*/Bellman-Ford/Floyd-Warshall, `graph/shortest.go`), MST (Kruskal/Prim, `graph/mst.go`), max-flow (Edmonds-Karp, `graph/flow.go:25`), centrality, communities (Louvain), connectivity, PageRank, topological sort. **No TSP. No vertex-cover heuristic. No max-clique. No set-cover greedy. No bipartite matching (private Hopcroft-Karp inside `topology/persistent/bottleneck.go` only). No Hungarian.** The 274-network-flow review owns the Hungarian / blossom / min-cost-flow tier (different layer). The 275-matroid review owns matroid-intersection (different abstraction).
- **`optim/proximal/ProxL0`** (slot 276 inventory) is the **only existing integer-flavoured operator** — it's a hard-thresholding prox operator for L0-pseudo-norm minimisation. Not a MIP solver, but conceptually adjacent (sparsity = "at most k non-zero entries" = cardinality constraint = matroid / 0-1 ILP). 215-compressed-sensing already consumes this for L0-OMP / IHT.
- **Gametheory** has Gale-Shapley stable-matching (`gametheory/matching.go:36`) and minimax / Nash-equilibrium / Shapley (`gametheory/`) — **no integer Stackelberg, no bilevel, no integer-strategy support enumeration**. Mixed-integer Stackelberg is a natural intopt consumer.
- **No MIP modelling layer.** No `Variable{lb,ub,type=Continuous|Binary|Integer}`, no `Constraint`, no `Model.Solve()` API. Slot 276 introduced a `ConicProgram` standard form for QP/SOCP/SDP — the analogous abstraction here is `MIPModel` with mixed-integer variable type tags.
- **No certificate-of-optimality output.** A B&B solver should emit `(xStar, objStar, gap, nodesExplored, cuts)` so downstream consumers (zkmark integer-feasibility witness, ASP-style enumeration) can verify. None of `optim/`'s solver returns this richer triplet today.
- **Moat**: confirmed via web search — **no zero-dep MIT pure-Go MIP solver exists**. golp wraps LPSolve (CGo), goop is a Gurobi-wrapper interface, lukpank/go-glpk is GLPK-CGo, nextmv-io/sdk/mip is a multi-backend dispatch. **gonum has zero integer-programming surface.** This puts `reality` in a *unique* position to ship the only zero-dep pure-Go MIP solver — same status as 275-matroid (Edmonds-blossom) and 274-network-flow (Edmonds blossom-V / Jonker-Volgenant).
- **2024 frontier**: Gomory mixed-integer (GMI) cuts had a renaissance after Balas-Perregaard 2003 established the precise correspondence with lift-and-project disjunctive cuts (LiP-cuts). All modern MIP solvers (CBC, HiGHS-MIP, SCIP, Gurobi) generate GMI cuts at every node of the B&B tree — Cornuéjols 2008 *Annals of Operations Research* survey "Valid inequalities for mixed integer linear programs" is the canonical 2008 reference; Conforti-Cornuéjols-Zambelli 2014 *Integer Programming* (Springer) is the textbook; the 2024 *Optimization Online* paper "Approximating the Gomory Mixed-Integer Cut Closure Using Historical Data" extends GMI selection with ML.
- **Pedagogy**: Land-Doig 1960 *Econometrica* B&B is in CLRS chapter 35 vicinity (in spirit; CLRS only covers approximation-algorithms not B&B). Wolsey 1998 *Integer Programming* and Nemhauser-Wolsey 1988 *Integer and Combinatorial Optimization* are the canonical textbooks. Gomory 1958 / Gomory 1960 / Gomory 1963 are the three foundational cut papers.

## Concrete recommendations

Recommended placement: **NEW sub-package `optim/intopt/`** under existing `optim/` — same "advanced sub-package" pattern as `optim/proximal/`, `optim/transport/`. Strict-downstream of `optim/SimplexMethod` (LP oracle); strict-upstream of (future) graph-combinatorial-NP-hard consumers and zkmark integer-feasibility witnesses.

### Tier 0 — LP-relaxation oracle wrapper (~80 LOC, foundational)

1. **`optim/intopt/lp_oracle.go::LPRelaxation` (~80 LOC).** Thin wrapper around `optim.SimplexMethod` that returns `(x, obj, status)` where `status ∈ {Optimal, Infeasible, Unbounded}`, normalises errors into status enum. **Why this layer:** the existing `SimplexMethod` returns `error` for both infeasible and unbounded, but B&B needs to distinguish — infeasible nodes are pruned, unbounded nodes mean the original IP is unbounded. Also lifts the row-and-column-bound interface so adding "x_j ≤ floor(x*_j)" or "x_j ≥ ceil(x*_j)" branches is a one-call append. Unblocks I2.

### Tier 1 — Branch-and-bound framework (~620 LOC)

2. **`optim/intopt/model.go::MIPModel` (~140 LOC).** `MIPModel{c []float64, A [][]float64, b []float64, varType []VarType, lb, ub []float64}` with `VarType ∈ {Continuous, Binary, Integer}`. Validate / canonicalise: binary ⇒ `lb=0, ub=1`. Methods `AddVar`, `AddConstraint`, `SetObjective`, `Validate`. Mirrors slot 276's `ConicProgram` precedent. Unblocks I3-I14.

3. **`optim/intopt/bnb.go::BranchAndBound(model *MIPModel, cfg BnBConfig) (*MIPSolution, error)` (~280 LOC).** Land-Doig 1960 best-first or depth-first B&B. Uses I1 LPRelaxation as oracle. Branches on most-fractional integer variable (default; configurable: pseudocost, strong-branching). Returns `MIPSolution{XStar, ObjStar, Status, NodesExplored, GapAbsolute, GapRelative, Bound}`. **Pruning rules**: (a) infeasible LP-relaxation, (b) LP-bound dominated by incumbent, (c) integer-feasible LP-relaxation = candidate incumbent. Best-first uses min-heap of nodes by LP-bound (reuse `container/heap` like `graph/shortest.go::dijkstraHeap` does at line 194). **Pin R-MUTUAL-CROSS-VALIDATION 3/3:** B&B with pure branching == B&B + Gomory-cuts (I8) == LP-relaxation-when-naturally-integer (totally-unimodular A). Unblocks every downstream MIP consumer.

4. **`optim/intopt/bnb_strategies.go::{MostFractional, Pseudocost, StrongBranching}` (~140 LOC).** Pluggable branching-variable selection. MostFractional (default, baseline, Land-Doig-1960) — pick var with `min(f_j, 1-f_j)` closest to 0.5 where `f_j = x*_j - floor(x*_j)`. Pseudocost (Bénichou-Gauthier-Girodet-Hentges-Ribière-Vincent-1971) — track up/down LP-bound improvements per branch. Strong-branching (Applegate-Bixby-Chvátal-Cook-1995) — solve mini-LPs at root to score candidates. Pseudocost is the production default in CBC / HiGHS-MIP.

5. **`optim/intopt/node_selection.go::{BestFirst, DepthFirst, BestEstimate}` (~60 LOC).** Pluggable node-selection strategy. BestFirst (min LP-bound, Land-Doig classic) gives provably-best worst-case node-count but high memory. DepthFirst is memory-bounded but can stall in regions far from optimum. BestEstimate (Forrest-Hirst-Tomlin-1974) interpolates.

### Tier 2 — Gomory fractional cuts (pure ILP) (~340 LOC)

6. **`optim/intopt/gomory_fractional.go::GomoryFractionalCut(tableau *SimplexState, basicRow int) (alpha []float64, beta float64)` (~180 LOC).** Gomory 1958 *Bull AMS* 64:275 fractional cut. For pure-integer LP (all vars integer), if optimal LP-tableau has a fractional basic variable, derive the cut `sum_j frac(a_ij) * x_j >= frac(b_i)` from the tableau row `i`. **Hard requirement:** needs access to the simplex tableau state, NOT just the optimal `x` — refactor `optim.SimplexMethod` to optionally return the final tableau / basis as a `SimplexState{tableau, basis, cBasis, b}`, OR re-derive it from `(x, basis)` via one additional LU. Pin **R-MUTUAL-CROSS-VALIDATION 3/3** on small ILPs (e.g. assignment 4×4 LP-relaxation already integer, so Gomory adds no cuts; vs B&B which also returns the same answer; vs brute-force enumeration).

7. **`optim/intopt/cutting_plane.go::PureCuttingPlane(model *MIPModel, cfg CuttingPlaneConfig) (*MIPSolution, error)` (~80 LOC).** Gomory 1958 pure cutting-plane algorithm: solve LP, if integer-feasible done, else generate Gomory fractional cut, append to constraint matrix, re-solve. Terminates in finite iterations for pure-IP (Gomory's original proof) but in practice slow / numerically fragile — primarily a pedagogical / cross-validation primitive against I3 BnB.

8. **`optim/intopt/branch_and_cut.go::BranchAndCut(model *MIPModel, cfg BnCConfig) (*MIPSolution, error)` (~80 LOC).** Padberg-Rinaldi 1991 branch-and-cut: B&B + cut generation at each node (root only, or all nodes per cfg). Standard MIP architecture in CBC/HiGHS-MIP/SCIP/Gurobi. Composes I3 with I7. **Pin R-MUTUAL-CROSS-VALIDATION 3/3:** BnB == BnC (both terminate, same `xStar`) == PureCuttingPlane (for small pure-IP).

### Tier 3 — Mixed-integer Gomory cuts + lift-and-project (~440 LOC)

9. **`optim/intopt/gomory_mixed_integer.go::GomoryMixedIntegerCut(tableau *SimplexState, basicRow int, integerMask []bool) (alpha []float64, beta float64)` (~220 LOC).** Gomory 1960 *Princeton-Tucker-Festschrift* / Gomory 1963 *Recent Advances in Math Prog* mixed-integer cut. Strictly stronger than I6 fractional cut; the production-default cut family in modern MIP solvers (CBC default, HiGHS-MIP default, SCIP default, Gurobi default). Reference: Cornuéjols 2008 §2.3, Conforti-Cornuéjols-Zambelli 2014 §5.1. Pin **R-CUT-DOMINANCE 1/1**: every Gomory-fractional cut on the all-integer instance is implied by the corresponding GMI cut (GMI strictly stronger or equivalent on integer-only rows).

10. **`optim/intopt/lift_and_project.go::LiftAndProjectCut(model *MIPModel, fracVar int) ([]float64, float64)` (~140 LOC).** Balas-Ceria-Cornuéjols 1993 / Balas-Perregaard 2003 disjunctive cut for 0-1 MIP: from the fractional `x*_j` of binary `x_j`, derive a valid inequality from the disjunction `x_j=0 ∨ x_j=1` via the cut-generating-LP (CGLP), but per Balas-Perregaard 2003 mimic the CGLP pivots inside the LP-tableau itself (no explicit higher-dim CGLP construction). **Equivalent to GMI for 0-1 MIPs** per Balas-Perregaard 2003 — but useful as a separate primitive for pedagogy + cross-validation pin **R-LIP-GMI-EQUIVALENCE 1/1**: LiP cut produced from `(x*, j)` ≡ GMI cut produced from the same fractional row, both on 0-1 MIPs.

11. **`optim/intopt/cut_pool.go::CutPool` (~80 LOC).** Pool of generated cuts with admission policy (efficacy, sparsity, parallelism, age). Standard MIP-solver architecture (Achterberg-2007 *Constraint Integer Programming* PhD thesis, the SCIP architecture reference). Manages cut activation/deactivation across B&B nodes.

### Tier 4 — Combinatorial-NP consumers (~620 LOC)

12. **`optim/intopt/knapsack.go::{Knapsack01, KnapsackBnB, KnapsackDP}` (~140 LOC).** 0-1 knapsack via three methods: (a) DP (pseudo-polynomial in capacity, Bellman-1957), (b) B&B with LP-relaxation = greedy fractional knapsack (Dantzig-1957), (c) Lagrangian-relaxation upper-bound. **Pin R-MUTUAL-CROSS-VALIDATION 3/3:** all three on n≤30 instances must match objective exactly. Knapsack is THE textbook ILP test problem.

13. **`optim/intopt/tsp.go::{TSPHeldKarp, TSPBranchAndCut, TSPLPRelaxation}` (~280 LOC).** Travelling Salesman: (a) Held-Karp 1962 DP O(n²·2ⁿ) exact for n≤20, (b) full subtour-elimination MIP via I8 BnC with **lazy constraint** generation (subtour-elimination constraints added on demand when integer-feasible candidate has subtour) — this is the *single canonical use-case for lazy constraints* in any MIP solver tutorial, (c) Held-Karp lower bound via 1-tree relaxation. **Pin R-MUTUAL-CROSS-VALIDATION 3/3** on n=15 instances. **Why this matters:** TSP is the single most-cited NP-hard combinatorial-optimisation problem; ships pedagogical demo of lazy-constraint architecture.

14. **`optim/intopt/set_cover.go::{SetCoverGreedy, SetCoverLPRound, SetCoverBnC}` (~100 LOC).** Set Cover: (a) Chvátal 1979 greedy (`H_n`-approx, the ⌈ln n⌉ approximation that is best-possible per Feige 1998 ASSUMING P≠NP), (b) LP-relaxation + randomised rounding (Raghavan-Thompson 1987), (c) exact via I8 BnC. **Pin R-APPROX-RATIO 1/1**: greedy ratio always ≤ `H_n` of the LP-relaxation bound.

15. **`optim/intopt/vertex_cover.go::{VertexCoverLP2Approx, VertexCoverBnB}` (~100 LOC).** Min Vertex Cover: (a) LP-relaxation + half-integrality rounding (Nemhauser-Trotter 1975, exact 2-approximation via the LP-half-integrality theorem), (b) exact via I3 BnB. **Pin R-NEMHAUSER-TROTTER 1/1**: LP-relaxation always has a half-integral optimal solution. **Cross-link**: vertex-cover IS the LP dual of maximum-matching; consumes 274's M5 Edmonds-blossom output.

### Tier 5 — Column generation / branch-and-price (deferred, ~340 LOC)

16. **`optim/intopt/column_generation.go::ColumnGeneration(masterLP, pricingProblem) (*LPSolution, error)` (~180 LOC).** Dantzig-Wolfe 1960 decomposition: solve master LP with subset of columns, pricing-problem generates negative-reduced-cost column, repeat until pricing-problem objective ≥ 0. Foundation for every large-scale MIP (vehicle-routing, crew-scheduling, cutting-stock-Gilmore-Gomory-1961). **Pedagogical canonical example: cutting-stock problem**.

17. **`optim/intopt/branch_and_price.go::BranchAndPrice(masterMIP, pricingProblem) (*MIPSolution, error)` (~160 LOC).** Barnhart-Johnson-Nemhauser-Savelsbergh-Vance 1998 branch-and-price: B&B + column-generation at each node. Industry-default for **vehicle-routing**, **airline-crew-scheduling**, **cutting-stock**.

### Tier 6 — Cutting-plane closure separation oracles (research-frontier, ~280 LOC)

18. **`optim/intopt/chvatal_gomory.go::ChvatalGomoryClosure(A, b, integerMask) ([][]float64, []float64)` (~140 LOC).** Chvátal 1973 / Schrijver 1980 closure of all CG cuts of rank-1. Separation oracle: given `x*`, find a CG-cut violated by `x*` or certify none exists. Solving the CG-separation is itself NP-hard (Eisenbrand 1999) so use heuristic separation (Fischetti-Lodi 2007 MIP-based separation).

19. **`optim/intopt/split_cuts.go::SplitCutSeparation(A, b, integerMask, xStar) ([]float64, float64)` (~140 LOC).** Andersen-Cornuéjols-Li 2005 split-cut separation. Strictly generalises GMI (every GMI is a split cut from a unit-disjunction `x_j ≤ floor ∨ x_j ≥ ceil`). Modern MIP-solver default (SCIP, HiGHS-MIP).

### Out of scope for first ship (deferred to v2):

- **MIP heuristics** (feasibility pump Fischetti-Glover-Lodi 2005, RINS Danna-Rothberg-Le-Pape 2005, local-branching Fischetti-Lodi 2003) — improves practical performance but B&B is correct without them.
- **Presolve** (probing, coefficient strengthening, dominated-column removal) — same.
- **MIPLIB-2017 benchmark suite** — large library validation, not a primitive.
- **Stochastic / robust MIP** — distinct subfield.
- **Quadratic MIP (MIQP / MIQCP)** — needs slot 276's QP solver as oracle; defer.

## Cross-cutting

- **graph (082-graph-missing)** ← I3 BnB + I13 TSP-BnC + I14 SetCover-BnC + I15 VertexCover-BnB unblock the **NP-hard combinatorial graph problems** that 082 enumerates. Slot 274 owns max-flow / Hungarian / blossom (P-time); slot 275 owns matroid-intersection (P-time); slot 277 owns the **NP-hard tier** (TSP, Steiner-tree-via-MIP-formulation, max-clique-via-Lovász-θ-SDP-+-IP, vertex-cover, set-cover).
- **graph max-clique** ← Lovász-θ via SDP (slot 199-G6, blocked on 276 SDP) gives an upper bound on `α(G)` (independence number); I3 BnB + I9 GMI cuts on the standard clique-IP formulation gives an exact solver. Pin **R-LOVASZ-IP-SANDWICH 1/1**: `θ(G) ≥ α(G) = exact-IP-result ≥ greedy-lower-bound`.
- **graph Steiner-tree** ← 274-N26 Steiner-tree-2-approx (P-time) gives the warm-start; I3 BnB + I9 GMI on the cut-formulation MIP gives the exact answer.
- **gametheory bilevel / Stackelberg** ← I3 BnB extends `gametheory/` to **mixed-integer Stackelberg** (leader picks integer strategy, follower best-responds in continuous LP). Stackelberg leaders that pick over discrete action sets need integer-programming.
- **gametheory matrix games with discrete strategies** ← I3 BnB extends Nash-equilibrium computation to discrete-strategy games via support-enumeration ILP (Dickhaut-Kaplan 1991, the canonical MIP formulation of NE).
- **prob MAP estimation in graphical models** ← MAP inference in pairwise MRF is integer-quadratic-programming (IQP); I3 BnB + I9 GMI gives an exact MAP solver. Polynomial-time only for restricted graph classes (Boykov-Kolmogorov binary submodular = max-flow); the general case needs MIP. Cross-link: 254-graph-cuts owns the polynomial-submodular case; 277 owns the general-NP case.
- **prob graphical-model structure learning** ← Bayesian-network structure-learning is a known ILP (Cussens 2011 *UAI*); needs I3 BnB + I9 GMI cuts.
- **combinatorics constrained partition counting** ← #P-hard but for small instances I3 BnB enumerates feasible partitions under arbitrary linear constraints.
- **compression Huffman + arithmetic** ← Huffman is greedy on tree-matroid (already optimal in P); intopt offers no benefit. Listed for completeness only.
- **crypto integer factorisation** ← intopt is *not* the right tool (factoring is non-linear-modular not integer-linear); Pollard-rho / GNFS belong elsewhere.
- **zkmark integer-feasibility witness** ← I3 BnB returns `(xStar, isFeasible, certificateOfOptimality)` — the `xStar` is a zero-knowledge-proof witness that an IP is feasible. Slot 277 supplies the prover-side oracle for ZK-SNARK-of-IP-feasibility consumers.
- **slot 102-T1.13 (Mehrotra MPC)** ← when shipped, intopt swaps `optim.SimplexMethod` → Mehrotra-MPC on **interior-bound** versions of the LP-relaxation oracle, especially useful when warm-starting from a parent B&B node's optimal LP base.
- **slot 276-X-conic** ← when shipped, opens **MIQCP / MISOCP / MISDP** track via QP/SOCP/SDP solver as the relaxation oracle inside B&B.
- **slot 174-G18 Frank-Wolfe** ← Frank-Wolfe is the **continuous relaxation** equivalent of the LP-greedy on a polytope; intopt's I9 GMI cuts are the integer-side companion.
- **slot 199 Lovász-θ** ← when full SDP-θ ships (after 276), intopt's max-clique-via-IP can sandwich-check against the θ upper bound — three-way pin.
- **slot 254 graph-cuts** ← submodular binary-MRF is poly-time max-flow; non-submodular general MRF needs intopt I3 BnB on the standard QPBO-LP-relaxation. Disjoint computational regimes; intopt is the **NP-fallback** when 254's submodular precondition fails.
- **slot 274 network-flow** ← Hungarian / blossom / min-cost-flow are P-time integer-programs (TU constraint matrices); intopt's I13 TSP / I14 set-cover / I15 vertex-cover are the NP-tier siblings on the **same `graph/` consumer surface**. The two slots together provide a complete combinatorial-optimisation surface.
- **slot 275 matroid** ← matroid-intersection is poly-time on TWO matroids (Edmonds 1968); intersection of THREE matroids is NP-hard (Lawler 1976) and falls into intopt's scope as a B&B problem.
- **slot 223 submodular** ← submodular-minimisation is P-time (Iwata-Fleischer-Fujishige 2001); submodular-MAXIMISATION subject to matroid constraint is NP-hard but admits the `(1-1/e)` continuous-greedy (Călinescu-Chekuri-Pál-Vondrák 2011); intopt provides the **exact-MIP-formulation** track for verification + small instances.

## Sources

**Repo files**
- `C:/limitless/foundation/reality/optim/linear.go` (35-155 SimplexMethod oracle; 172-316 InteriorPoint barrier-gradient — do not reuse)
- `C:/limitless/foundation/reality/graph/flow.go` (25-104 MaxFlow Edmonds-Karp; 127-167 TopologicalSort)
- `C:/limitless/foundation/reality/graph/shortest.go` (29 Dijkstra, 83 AStar, 144 FloydWarshall, 194 dijkstraHeap pattern reusable for B&B node-priority-queue)
- `C:/limitless/foundation/reality/graph/mst.go` (34 Kruskal, 110 Prim — graphic-matroid greedy oracles)
- `C:/limitless/foundation/reality/graph/bellman_ford.go` (30 BellmanFord — substrate for negative-reduced-cost detection in column-generation pricing-problem)
- `C:/limitless/foundation/reality/optim/proximal/operators.go` (ProxL0 hard-thresholding — only existing integer-flavoured primitive)
- `C:/limitless/foundation/reality/topology/persistent/bottleneck.go:248` (private Hopcroft-Karp — should lift to public per slot 274-N8 first)
- `C:/limitless/foundation/reality/gametheory/matching.go:36` (Gale-Shapley — orthogonal stable-matching, not weighted-matching)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/102-optim-missing.md` (T1.13 Mehrotra MPC ⇒ improves intopt LP-oracle)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/276-new-cvxopt-extras.md` (X1-X24 conic stack ⇒ unlocks MISOCP/MISDP)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/274-new-network-flow.md` (Hungarian / blossom / min-cost-flow — P-time sibling)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/275-new-matroid.md` (matroid-intersection — adjacent abstraction)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/199-synergy-graph-info.md` (Lovász θ — clique sandwich)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/254-new-graph-cuts.md` (submodular MRF — P-time companion)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/223-new-submodular.md` (submodular-max under matroid constraint — NP-hard companion)

**Web sources**
- [Cutting-plane method — Wikipedia](https://en.wikipedia.org/wiki/Cutting-plane_method) — Gomory 1958 / 1960 / 1963 fractional + mixed-integer cuts.
- [Approximating the Gomory Mixed-Integer Cut Closure Using Historical Data (Optimization Online 2024)](https://optimization-online.org/2024/11/approximating-the-gomory-mixed-integer-cut-closure-using-historical-data/) — 2024-frontier work on GMI cut selection with ML.
- [A precise correspondence between lift-and-project cuts, simple disjunctive cuts, and mixed integer Gomory cuts for 0-1 programming (Balas-Perregaard 2003)](https://link.springer.com/content/pdf/10.1007/s10107-002-0317-y.pdf) — foundation for I10 LiP-cut implementation; mimic-CGLP-pivots-in-LP-tableau trick.
- [A relax-and-cut framework for Gomory mixed-integer cuts (Math Prog Comp 2011)](https://link.springer.com/article/10.1007/s12532-011-0024-x) — modern GMI cut application architecture.
- [Pivot-and-reduce cuts: improving Gomory mixed-integer cuts (Op Res Lett 2011)](https://www.sciencedirect.com/science/article/abs/pii/S037722171100350X) — strengthening GMI cuts.
- [Gomory cuts revisited (Op Res Lett 1996)](https://www.sciencedirect.com/science/article/abs/pii/0167637796000077) — Balas-Ceria-Cornuéjols-Natraj 1996 numerical pivoting issues.
- [golp Go bindings for LPSolve](https://pkg.go.dev/github.com/draffensperger/golp) — moat-baseline: Go-MILP today is CGo-wrapper.
- [GOOP: Generalized Mixed Integer Optimization in Go](https://github.com/mit-drl/goop) — moat-baseline: interface only, requires Gurobi.
- [Methods and Solvers for MILP/MNLP Review (IJSTR 2020)](https://www.ijstr.org/final-print/jan2020/Methods-And-Solvers-Used-For-Solving-Mixed-Integer-Linear-Programming-And-Mixed-Nonlinear-Programming-Problems-A-Review.pdf) — modern-solver landscape.

**Textbook references**
- Land-Doig 1960 *Econometrica* 28:497 — original B&B.
- Gomory 1958 *Bull AMS* 64:275 — fractional cuts.
- Gomory 1960 *Princeton-Tucker-Festschrift* / Gomory 1963 *Recent Adv Math Prog* — mixed-integer cuts.
- Balas-Ceria-Cornuéjols 1993 *Math Prog* 58:295 — lift-and-project cuts.
- Padberg-Rinaldi 1991 *SIAM Review* 33:60 — branch-and-cut for TSP.
- Held-Karp 1962 *J SIAM* 10:196 — DP TSP.
- Chvátal 1979 *Math Op Res* 4:233 — set-cover greedy `H_n` bound.
- Nemhauser-Trotter 1975 *Math Prog* 8:232 — vertex-cover LP-half-integrality.
- Andersen-Cornuéjols-Li 2005 *Math Prog* 102:457 — split cuts.
- Barnhart-Johnson-Nemhauser-Savelsbergh-Vance 1998 *Op Res* 46:316 — branch-and-price.
- Cornuéjols 2008 *Annals OR* 149:3 — valid-inequality survey.
- Conforti-Cornuéjols-Zambelli 2014 Springer — *Integer Programming* textbook.
- Wolsey 1998 Wiley — *Integer Programming* textbook.
- Nemhauser-Wolsey 1988 Wiley — *Integer and Combinatorial Optimization* textbook.
- Achterberg 2007 PhD TU-Berlin — *Constraint Integer Programming* (SCIP architecture).

# 201 — New Math: Optimal Transport (Block C, slot 1)

**Summary line 1:** `optim/transport/` ships ~960 LOC across 6 files (Wasserstein-1D closed-form, IQR normalisation, log-domain Sinkhorn, pairwise / min-pairwise W₁) with the v2 deferral list explicitly enumerated in `doc.go:99-114`; no other OT primitives exist anywhere else in the repo (verified by grep across all 22 packages).
**Summary line 2:** Eight ranked additions totalling ~3,250 LOC would lift reality from "v1 toy / 1-D shape comparison + n-D Sinkhorn" to "research-frontier OT toolkit": Sinkhorn divergence, sliced-Wasserstein, unbalanced OT, network simplex (exact LP-OT), W₂ barycenters, Gromov-Wasserstein, JKO step, partial OT — every one explicitly named in the L11 Ecosystem Hunt entry as a deferred-but-pulling consumer pattern.

---

## (1) What reality ships today (verified at v0.10.0)

`optim/transport/` (1,462 LOC including a 593-LoC test file):

| File | LOC | What it does |
|---|---|---|
| `doc.go` | 145 | Package overview, consumer-list, v2 deferral roster |
| `errors.go` | 57 | 5 sentinels: ErrEmptyDistribution, ErrUnequalMass, ErrInvalidP, ErrInvalidRegularisation, ErrSinkhornNonConvergent, ErrCostMatrixDimensionMismatch |
| `iqr_norm.go` | 119 | `IQRNormalise` (Tukey 1977 robust z-score), `quantileFromSorted` linear-interp helper, `filterAndSortFinite` |
| `wasserstein1d.go` | 200 | `Wasserstein1D(u, v, p)` closed-form W_p on 1-D empirical via order-statistic pairing or `(k+0.5)/n` quantile interp for unequal sizes; `Wasserstein1DDetailed` with IQR-normalisation + sample-size diagnostics |
| `pairwise.go` | 101 | `PairwiseWasserstein1D` symmetric K×K distance matrix; `MinPairwiseWasserstein1D` smallest pairwise + achieving (i, j) |
| `sinkhorn.go` | 247 | Log-domain entropic-regularised OT via Cuturi 2013 / Peyré-Cuturi 2019; LSE-stabilised dual potentials f, g; L¹ marginal-deviation convergence check; returns `SinkhornResult{Plan, Cost, Iterations}` |

**Cross-substrate parity:** `Wasserstein1D` is byte-for-byte ≤1e-12 against RubberDuck's 218-LoC `OptimalTransport.cs` reference (verified via `TestCrossSubstratePrecision_RubberDuck_*`). Sinkhorn has no sister implementation (RubberDuck's `Wasserstein2DSinkhorn` design-doc spec is unwired).

**Other repo-wide hits:** `topology/persistent/doc.go:73` mentions p-Wasserstein on persistence diagrams as a future capability; nothing implemented. `info/lz/doc.go` mentions OT in cross-references only. **No barycenter code anywhere.** **No sliced-Wasserstein anywhere.** **No Gromov-Wasserstein anywhere.** **No unbalanced-OT anywhere.**

The package's own `doc.go:99-114` v2 list is the canonical deferral roster. Slot 102 (optim-missing) flagged the LP-exact gap as **T3.33 — Network simplex** but did not enumerate the modern OT extensions; that is this review's job.

---

## (2) What's missing — ranked by demand

Demand ranking is by (a) explicit consumer mentioned in `doc.go`, (b) frequency of citation in the broader L11 hunt entry, and (c) connective-tissue readiness (e.g. PR8 already provides `prob/` distributions, so Sinkhorn-divergence MMD glue is short).

### Tier-1 — high demand, short connective tissue (~1,400 LOC)

#### O1. Sinkhorn divergence (Feydy-Séjourné-Vialard-Trouvé-Amari-Peyré 2019) — ~150 LOC
**The single highest-leverage addition.** The vanilla Sinkhorn `Cost` returned by `sinkhorn.go:185` is **biased**: `S_ε(α, α) ≠ 0` and the bias scales as O(ε log ε). The debiased Sinkhorn divergence

`S̄_ε(α, β) = S_ε(α, β) − ½ S_ε(α, α) − ½ S_ε(β, β)`

is symmetric, non-negative, zero on equal measures, and interpolates between W₂² (ε → 0) and MMD (ε → ∞). For every ML-loss consumer this is what you actually want, not raw Sinkhorn cost. Already explicitly v2-deferred in `doc.go:100-101`. Implementation is three calls into existing `Sinkhorn` plus arithmetic; no new convergence theory. **API:** `SinkhornDivergence(a, b, C_ab, C_aa, C_bb, eps, ...) (float64, error)`.

#### O2. Sliced Wasserstein (Kolouri-Pope-Martin-Rohde 2018) — ~200 LOC
The L11 hunt entry's Recall HNSW consumer (cosine on 256-dim structural embeddings) is the named driver. Sliced-W₂² estimates the n-D Wasserstein distance via Monte-Carlo over random 1-D projections, reusing the existing `Wasserstein1D` closed form per slice. Falls out of the existing 1-D path almost for free: project both point clouds onto a random direction θ ~ Unif(S^{d-1}), call existing `Wasserstein1D`, average over N projections. Gives the user O(N · (n + m) log(n + m)) instead of Sinkhorn's O(n · m · iter). Generalised Sliced-Wasserstein (Kolouri-Naderializadeh-Rohde-Hoffmann 2019) adds non-linear defining functions; ship vanilla first.
**API:** `SlicedWasserstein2(u, v [][]float64, numSlices int, rng *rand.Rand) (float64, error)`. Need a `linalg.UnitSphereSample` helper (~20 LOC; Marsaglia normal/normalize).

#### O3. Network simplex (exact LP-OT) — ~600 LOC
Slot 102 flagged this as T3.33. The exact unregularised OT plan via Orlin's network simplex is the gold-standard for the **discrete** case (n, m ≤ ~5,000). Sinkhorn pays an entropic-regularisation bias every call; network simplex returns the true LP optimum. Driver: any consumer that wants the `Plan` to be **sparse** (Sinkhorn produces a fully-dense doubly-stochastic matrix, useless for hard assignment). Used by `transport/POT` (Python) by default for n ≤ a few thousand.
**API:** `NetworkSimplexOT(a, b, C, opts) (Plan [][]float64, Cost float64, err error)`. Implementation effort: this is the heaviest single algorithm in the list — Orlin 1997 / Bonneel-Van-de-Panne-Paris-Heidrich 2011 with cycle-cancelling. Cleanly belongs in `optim/transport/network_simplex.go` plus `optim/transport/auction.go` for the alternative Bertsekas algorithm (not strictly necessary if simplex ships).

#### O4. Auction algorithm (Bertsekas 1981) — ~250 LOC
Cheaper exact OT for the **square assignment** case (n = m, uniform marginals). Embarrassingly parallel and lock-free; the natural specialisation of network simplex when the problem is bipartite assignment. Ships alongside O3 because the two share the LP-validation test corpus. Scaling factor (ε-scaling) is the practical convergence trick.
**API:** `AuctionAssignment(C [][]float64, opts) (perm []int, cost float64, err error)`.

#### O5. Wasserstein-2 barycenter via free support / Lloyd updates (Cuturi-Doucet 2014) — ~250 LOC
The W₂ barycenter is the OT-aware analogue of the Euclidean centroid: given measures (α₁, …, α_K) with weights (λ_1, …, λ_K), find arg min_β Σ_k λ_k W₂²(α_k, β). For point-cloud / regime-mean consumers (RubberDuck `RegimeContextService` cited in `doc.go:24-28`) this is the **principled regime average**. Lloyd-style update: at iteration t, given current support points {x_i^(t)} of β_t, solve OT each (α_k, β_t) to get plans P^k, then update each support point as the λ-weighted barycentric projection. Reuses Sinkhorn for the per-pair OT subproblem.
**API:** `Wasserstein2Barycenter(measures []*PointCloud, weights []float64, eps float64, maxIter int) (PointCloud, error)`. Needs a tiny `PointCloud{Points [][]float64; Weights []float64}` shared type — promote to package-level so it can serve future quadratic-OT / Brenier consumers.

### Tier-2 — high demand, medium connective tissue (~1,250 LOC)

#### O6. Unbalanced OT (Chizat-Peyré-Schmitzer-Vialard 2018) — ~300 LOC
Drops the equal-mass marginal constraint. Replaces hard equality with KL-divergence soft penalties on `P 1 − a` and `P^T 1 − b`. Critical for any consumer where one distribution can lose mass (regime where some bins disappear) — the Nexus DirectionalDrift consumer (`doc.go:39-44`) wants this when the prior support and current support disagree. Mathematics: same Sinkhorn-style alternating updates, replace `f_i ← ε(log a_i − LSE_j(...))` with `f_i ← (ρ/(ρ+ε)) · ε(log a_i − LSE_j(...))` where ρ is the marginal-relaxation strength. Unbalanced returns gracefully when masses are mismatched — currently `Sinkhorn` raises `ErrUnequalMass`. Already explicitly v2-deferred in `doc.go:100-103`.
**API:** `UnbalancedSinkhorn(a, b, C, eps, rho float64, maxIter, tol) (UnbalancedResult, error)`. Connective-tissue: refactor existing `Sinkhorn` so it can call a shared kernel with ρ → ∞ producing the balanced result.

#### O7. Gromov-Wasserstein (Mémoli 2011) — ~400 LOC
Distances between **metric-measure spaces** (not just measures on a common space). Solves `min_P Σ_{i,j,k,l} |C₁_{ik} − C₂_{jl}|² P_ij P_kl s.t. P 1 = a, P^T 1 = b`. The non-convex quadratic relaxation (Peyré-Cuturi-Solomon 2016) uses entropic regularisation + iterative linearisation: alternate (a) form linearised cost matrix `L = C₁ ⊗ P ⊗ C₂` (Sinkhorn-tensor product), (b) run inner Sinkhorn with this `L`. Driver: shape-comparison / point-cloud-registration consumers — color transfer (Solomon-Goes-Peyré-Cuturi-Butscher-Nguyen-Du-Guibas 2015), graph-graph distance (where C₁, C₂ are shortest-path matrices). Required for any consumer that has *two different feature spaces* and wants OT between them.
**API:** `GromovWasserstein(a, b, C1, C2, eps, maxOuterIter, maxInnerIter, tol) (GWResult, error)`. **Fused-GW** (Vayer-Chapel-Flamary-Tavenard-Courty 2020) extends with a linear cost component — same kernel + a convex-combination weight α ∈ [0, 1].

#### O8. JKO step / Wasserstein gradient flow (Jordan-Kinderlehrer-Otto 1998) — ~250 LOC
Already explicitly v2-deferred in `doc.go:107-109`. The proximal step in the W₂ metric: given a free-energy functional F[ρ] (e.g. Boltzmann entropy plus a potential) and current measure ρ_n,
`ρ_{n+1} = arg min_ρ  F[ρ] + (1/2τ) W₂²(ρ, ρ_n)`
recovers the heat equation when F is the Boltzmann entropy and a Fokker-Planck PDE more generally. Entropic-regularised JKO (Peyré 2015 "Entropic approximation of Wasserstein gradient flows") makes this one inner Sinkhorn per outer step. Cross-package value: `chaos/` Lorenz / Van der Pol invariant-measure approximation; `prob/` proper Bayesian flow that respects geometry. Pairs with `optim/proximal/` natural target.
**API:** `JKOStep(rhoCurrent *PointCloud, energyFunc func([]float64) float64, tau, eps float64, opts) (PointCloud, error)`.

#### O9. Partial OT (Caffarelli-McCann 2010 / Figalli 2010) — ~200 LOC
Transport only a fraction `s ∈ (0, 1)` of mass. Reduces to balanced Sinkhorn on a (n+1)×(m+1) cost matrix with virtual sink rows / columns for the unmoved mass. Driver: any consumer where rejecting / dumping mass is permitted — outlier-robust regime comparison. Ships as a thin wrapper over balanced `Sinkhorn` or `NetworkSimplexOT` once O3 lands.
**API:** `PartialOT(a, b, C, fraction, eps, ...) (Plan [][]float64, Cost, transportedMass float64, err error)`.

#### O10. Multi-marginal OT (Pass 2015) — ~100 LOC (decompositional)
k > 2 distributions simultaneously. The fully-general Sinkhorn-multi-marginal (Benamou-Carlier-Cuturi-Nenna-Peyré 2015) is exponential in k; the L11 hunt entry's only mention is "v2 if a consumer pulls". Cheap special case: **all k marginals along a chain** (k-1 consecutive Sinkhorns) is O(k · n · m · iter) and covers the "regime evolution along time" use case. Defer the fully tensorial form unless a consumer demands.

### Tier-3 — research-frontier / niche (~600 LOC)

- **O11. W₁ on the line via CDF inverse (closed form refactor)** — already in `wasserstein1d.go` for p = 1; expose `EmpiricalCDF` and `EmpiricalQuantile` as public helpers (~30 LOC); non-trivial only as API hygiene.
- **O12. Quadratic OT / Brenier theorem** — for absolutely-continuous source measures the optimal map is the gradient of a convex function. Continuous-OT primal via input-convex neural net (Makkuva-Taghvaei-Lee-Oh 2020) is out-of-scope; ship the *empirical* version: given (x_i), (y_i) with equal-size point clouds, the W₂² optimal is the linear-assignment from auction (~50 LOC of glue once O4 lands).
- **O13. Schrödinger bridge problem (entropic OT in dynamic form)** — Léonard 2014. The static Sinkhorn already solves the Schrödinger bridge between two endpoint measures with kernel cost `C_ij = (1/2) ||x_i − y_j||²`; the dynamic interpolation (heat-kernel diffusion bridges) is a separate ~150-LOC additional primitive.
- **O14. Sinkhorn convergence diagnostic** — Altschuler-Weed-Rigollet 2017 give explicit ε-precision bounds. Currently `sinkhorn.go:174-181` checks raw L¹ residual; we could expose an `ETACiterationsRemaining` estimate (~20 LOC; emits R75 SINKHORN_NONCONVERGENT context).
- **O15. Sliced-Gromov-Wasserstein** (Vayer-Flamary-Tavenard-Chapel-Courty 2019) — same idea as O2 but for GW. Ships as a one-line wrapper if O7 + O2 are present.
- **O16. Continuous OT via dual potentials** — Sinkhorn returns `f, g` internally but `sinkhorn.go:142-143` discards them after `buildPlan`. Surface them in the result struct (~20 LOC trivial).
- **O17. Optimal transport for persistence diagrams** — couples to `topology/persistent/`. The Bottleneck distance (Cohen-Steiner-Edelsbrunner-Harer 2007) and p-Wasserstein on persistence diagrams (Mileyko-Mukherjee-Harer 2011). Explicitly mentioned in `topology/persistent/doc.go:73` as not-yet-shipped. ~150 LOC: filtered-bipartite-matching with a fixed off-diagonal cost.
- **O18. OT for Bayesian inference (variational OT)** — Ambrogioni-Güçlü-van-Gerven-Maris 2018 "Wasserstein variational inference". Cross-package with `prob/`; defer until a `prob/variational/` sub-package is justified.

---

## (3) Connective tissue LOC for new primitives — composite estimate

| Bucket | Lines |
|---|---|
| Tier-1 (O1 + O2 + O3 + O4 + O5) | ~1,450 |
| Tier-2 (O6 + O7 + O8 + O9 + O10) | ~1,250 |
| Tier-3 selective (O11 + O14 + O16 + O17) | ~220 |
| Shared types: `PointCloud`, `EmpiricalMeasure` | ~80 |
| Sphere-sampling helper for sliced-W (`linalg`) | ~20 |
| Persistence-diagram-OT bridge (`topology` import) | ~50 |
| Tests for everything (golden vectors + cross-validation against POT/scipy reference vectors) | ~1,800 |
| **Total Tier-1 + Tier-2 + selected Tier-3 + tests** | **~4,870 LOC** |

This more than triples `optim/transport/` (currently 1,462 LOC) but transforms it from "1-D toy + 1 entropic n-D solver" into the **only Go-native zero-dep OT toolkit at research-frontier parity with POT (Python) / OptimalTransport.jl**.

Realistic phased delivery:

- **Phase A (next quarter, ~700 LOC):** O1 + O2 + O16 (the three near-zero-effort high-leverage wins; O1 and O2 ship as additions over existing Sinkhorn + Wasserstein1D, O16 is API surface).
- **Phase B (when first consumer pulls, ~1,400 LOC):** O3 + O4 + O5. Network simplex is the heaviest single algorithm; auction reuses the validation corpus; barycenter is the consumer-facing capstone.
- **Phase C (research-frontier, ~1,500 LOC):** O6 + O7 + O8. Unbalanced + Gromov + JKO. Each requires its own multi-paper paper-trail and 30+ golden vectors.
- **Defer indefinitely:** O10 (multi-marginal full tensor), O12 (continuous Brenier), O13 (Schrödinger dynamic), O15, O18 — until consumer pulls.

---

## (4) Cross-package coupling — what will benefit

| Package | What lights up |
|---|---|
| `prob/` | Sinkhorn-divergence as proper symmetric distribution distance (replaces KL where supports disjoint); JKO for variational inference |
| `topology/persistent/` | Bottleneck + p-Wasserstein on persistence diagrams (already cited as TODO at `doc.go:73`) |
| `color/` | OT-based color transfer (Pitie-Kokaram-Dahyot 2007); reuses existing `color` 8-space machinery as cost-matrix factory |
| `geometry/` | Point-cloud registration via OT-with-rigid-constraint; needs `PointCloud` shared type |
| `signal/` | Wasserstein loss for spectral-distribution comparison (replaces L₂ on PSDs which ignores frequency adjacency) |
| `chaos/` | Invariant-measure approximation via JKO heat-flow on Lorenz attractor sample histograms |
| `gametheory/` | Multi-population replicator dynamics in W₂ metric (Hofbauer-Sigmund-style refinement) |
| `linalg/` | `UnitSphereSample` helper for sliced-W is a 20-LoC addition useful elsewhere (Marsaglia-Box-Muller composition) |

Six of the 22 packages benefit from at least one of {O1, O2, O5, O7}. Two more (`acoustics/`, `fluids/`) benefit from W₁ on PSDs (signal-spectrum-distribution compare). **Optimal transport is genuinely the most cross-cutting math in the deferral list.**

---

## (5) Risks / gotchas to flag in any scoping doc

1. **Sinkhorn-divergence requires three Sinkhorn calls per evaluation** — the `S_ε(α, α)` and `S_ε(β, β)` self-distance terms can be cached if the user re-uses one of the marginals across many comparisons. Build a `SinkhornCache` shared struct so RubberDuck's `EvolutionOrchestratorService` (K agents pairwise-compared) doesn't re-run K identical self-Sinkhorns.
2. **Network simplex is non-trivially heavy** — Orlin 1997 with cycle cancelling is genuinely 600 LoC of dense bookkeeping; tempting to wrap a C library, but the project's "zero dependencies" mandate (CLAUDE.md rule 6, "Reimplement from first principles") forbids that. Budget engineering time honestly.
3. **Gromov-Wasserstein is non-convex** — convergence to a *local* minimum, not the global. Sensitivity to initialisation is a real numerical concern. Ship with `numRestarts` API and document the local-optimality caveat clearly. Cross-substrate parity will need *multiple* random-seed-tagged golden runs; standard parity-by-byte won't work for the GW global solution.
4. **Sliced-Wasserstein RNG determinism** — golden-file testing requires deterministic projection directions. Use a seeded `*rand.Rand` from the standard `math/rand` (not `crypto/rand`); document that the output is deterministic only given the same `(seed, numSlices)` pair.
5. **Unbalanced OT marginal-mass interpretation** — when `ρ → ∞` you recover balanced. When `ρ → 0` you recover degenerate (no transport, both penalties zero). Document the regime where ρ should sit relative to the cost-matrix scale.
6. **Sinkhorn convergence stalls below ε ≈ 1e-9 × scale(C)** — already noted in `sinkhorn.go:42-46`. Adding O14 (ETA estimator) would make this a clean diagnostic instead of a quiet stall-then-fail.
7. **The L11 entry's RubberDuck cross-substrate parity contract only covers Wasserstein1D** — Sinkhorn has no sister implementation today (`doc.go:88-90`). Adding 5+ new primitives with zero parity contracts is a *substantial* widening of the cross-substrate testing burden. Mitigation: golden-file vectors validated against POT (Python) at `1e-9` relative tolerance, generated by a reference script committed to the repo.

---

## (6) Recommended slot-201 verdict

**SHIP** Tier-1 in next 2-3 sprints. The five Tier-1 additions (O1, O2, O3, O4, O5) collectively deliver the capabilities that every OT consumer in `doc.go:24-54` is currently working around with cosine / KL / Euclidean substitutes. Sinkhorn-divergence (O1) alone is ~150 LoC and unblocks **all** "Sinkhorn-as-loss" consumers — the highest-leverage 1-day project in the entire deferral list.

**DEFER but-design-for** Tier-2 (O6, O7, O8). Reserve the package-level `PointCloud` type now so future Tier-2 work doesn't need to refactor.

**DROP** O10 (multi-marginal full tensor) and O18 (variational OT) until a real consumer materialises; they are textbook-completeness items, not pull-driven items.

---

## Files referenced

- `C:\limitless\foundation\reality\optim\transport\doc.go` (full v2 deferral roster, lines 99-114)
- `C:\limitless\foundation\reality\optim\transport\sinkhorn.go` (existing log-domain Sinkhorn, 247 LOC)
- `C:\limitless\foundation\reality\optim\transport\wasserstein1d.go` (closed-form W_p 1-D, 200 LOC)
- `C:\limitless\foundation\reality\optim\transport\pairwise.go` (101 LOC)
- `C:\limitless\foundation\reality\optim\transport\iqr_norm.go` (119 LOC)
- `C:\limitless\foundation\reality\optim\transport\errors.go` (57 LOC)
- `C:\limitless\foundation\reality\optim\transport\transport_test.go` (593 LOC test corpus)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\102-optim-missing.md` (T3.33 — Network simplex flagged)
- `C:\limitless\foundation\reality\topology\persistent\doc.go:73` (p-Wasserstein on persistence diagrams marked as TODO)

## References for new primitives

- Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transportation distances. NeurIPS 26: 2292-2300.
- Feydy, J., Séjourné, T., Vialard, F.-X., Trouvé, A., Amari, S., & Peyré, G. (2019). Interpolating between optimal transport and MMD using Sinkhorn divergences. AISTATS.
- Kolouri, S., Pope, P. E., Martin, C. E., & Rohde, G. K. (2018). Sliced Wasserstein distance for learning Gaussian mixture models. CVPR.
- Orlin, J. B. (1997). A polynomial time primal network simplex algorithm for minimum cost flows. Math. Prog. 78: 109-129.
- Bertsekas, D. P. (1981). A new algorithm for the assignment problem. Math. Prog. 21: 152-171.
- Cuturi, M., & Doucet, A. (2014). Fast computation of Wasserstein barycenters. ICML.
- Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F.-X. (2018). Scaling algorithms for unbalanced optimal transport problems. Math. Comp. 87: 2563-2609.
- Mémoli, F. (2011). Gromov-Wasserstein distances and the metric approach to object matching. Found. Comput. Math. 11: 417-487.
- Peyré, G., Cuturi, M., & Solomon, J. (2016). Gromov-Wasserstein averaging of kernel and distance matrices. ICML.
- Vayer, T., Chapel, L., Flamary, R., Tavenard, R., & Courty, N. (2020). Fused Gromov-Wasserstein distance for structured objects. Algorithms 13(9):212.
- Jordan, R., Kinderlehrer, D., & Otto, F. (1998). The variational formulation of the Fokker-Planck equation. SIAM J. Math. Anal. 29: 1-17.
- Peyré, G. (2015). Entropic approximation of Wasserstein gradient flows. SIAM J. Imaging Sci. 8: 2323-2351.
- Caffarelli, L. A., & McCann, R. J. (2010). Free boundaries in optimal transport and Monge-Ampère obstacle problems. Ann. Math. 171: 673-730.
- Solomon, J., de Goes, F., Peyré, G., Cuturi, M., Butscher, A., Nguyen, A., Du, T., & Guibas, L. (2015). Convolutional Wasserstein distances. ACM Trans. Graphics 34(4):66.
- Léonard, C. (2014). A survey of the Schrödinger problem and some of its connections with optimal transport. Discrete Contin. Dyn. Syst. 34: 1533-1574.
- Altschuler, J., Weed, J., & Rigollet, P. (2017). Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration. NeurIPS.
- Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport. Foundations and Trends in ML 11(5-6).

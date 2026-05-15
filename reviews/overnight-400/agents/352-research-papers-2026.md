# 352 — research-papers-2026 (2026 YTD papers reality should track)

## Headline
2026 YTD (Jan–May, ~4 months) is sparser than 2025 but produced concrete advances in shortest-path graph theory, GPU-stable Sinkhorn OT, sum-check prover speedups, multiparameter persistence dualities, manifold-aware Kalman filtering, and several incremental refinements to randomized NLA / conformal prediction. Reality should pull ~6–8 of these and largely *defer* the rest until the Sept 2026 corpus closes.

## Caveat — corpus depth
With only ~4 months of arxiv output, many "2026" hits are either v2 revisions of 2025 work or routine extensions. The shortest-path / persistence / sum-check items below are genuinely new in 2026; most others are best understood as continuations of the themes catalogued by slot 351.

## Top papers (sorted by impact × fit)

### 1. Shortest-paths "sorting barrier" follow-on — Linear-time SSSP for Euclidean graph classes (arxiv:2603.22948, Mar 2026)
- Result: Building on the 2025 sorting-barrier breakthrough for directed SSSP (arxiv:2504.17033), proves linear-time single-source shortest path for any graph class whose contracted graphs admit sublinear separators, with explicit instantiations for planar / Euclidean cases. Companion paper "Faster shortest-paths via the acyclic-connected tree" (arxiv:2504.08667) generalizes the SCC decomposition.
- Reality slot: `graph.SSSPEuclidean` and `graph.ACTree` decomposition primitive; complements existing Dijkstra/A*.
- Pure-Go feasibility: ~600 LOC for ACTree; Euclidean separator routine adds ~400 LOC. No external deps.
- Recommend track? **Y, prefer.** Reality already has Dijkstra; this is the first sub-Dijkstra SSSP since 1984 for the relevant graph class. Foundational.

### 2. Mao et al. 2026 — New Greedy Spanners and Applications (arxiv:2603.17085)
- Result: Simple greedy procedure outputs (k, k−1)-multiplicative spanners with f-edge fault tolerance of size O(f n^{1+1/k}) — the first tight result for non-multiplicative fault-tolerant spanners.
- Reality slot: new `graph.Spanner` (none currently in reality).
- Pure-Go feasibility: ~300 LOC; just sorted edge greedy + Floyd-Warshall-style distance check.
- Recommend track? **Y.** Small, self-contained, broadly useful for network-analysis users.

### 3. FastSinkhorn — Log-domain Sinkhorn with Warp-Level Reductions (arxiv:2605.00837, May 2026)
- Result: Pure log-domain Sinkhorn stable at ε = 10⁻⁴; on n = 8192, 12× faster than POT on CPU baseline; numerical stability prescription is the load-bearing contribution (the GPU kernels are not).
- Reality slot: `prob.Sinkhorn` or new `optim.OT` — reality has no OT solver.
- Pure-Go feasibility: log-domain CPU version ~250 LOC. Skip the CUDA part; the numerical recipes (max-stabilization, log-sum-exp ordering) port verbatim.
- Recommend track? **Y, prefer.** Replaces / forecloses the bare-Sinkhorn versions that overflow at small ε.

### 4. Speeding Up Sum-Check Proving (eprint 2026/587)
- Result: Three complementary techniques reducing sum-check prover time and memory, with measured wins inside zk-VM workloads (~30–60% prover-time reduction). Composes with HyperPlonk / Nova frontends.
- Reality slot: extension to the slot-351-recommended `crypto/zk/sumcheck`.
- Pure-Go feasibility: ~250 LOC delta; algorithmic, no new field arithmetic needed.
- Recommend track? **Y.** If reality adopts sum-check at all, take this version.

### 5. ∆-SQIsign — Degree-Challenge Isogeny Signature (eprint 2026/852) + Survey of Castryck-Decru-resistant schemes (eprint 2026/446)
- Result: ∆-SQIsign is a new SQIsign variant whose challenge is the *degree* of an isogeny; survey paper rationalizes the post-Castryck-Decru landscape (SQIsign, SQIsign2D-{East,West}, SQIsignHD, SQIPrime) and their security tradeoffs.
- Reality slot: future `crypto/sqisign` track only — too early to ship.
- Pure-Go feasibility: ~5000 LOC for full SQIsign, including supersingular endomorphism rings, KLPT, quaternion algebra. Heavy.
- Recommend track? **Y, document only.** ML-DSA + ML-KEM (slot 351 paper 4) remain the actionable PQC bets; isogenies stay in the watch-list.

### 6. Krishnan et al. 2026 — Dualities in Multiparameter Persistence (arxiv:2603.18224)
- Result: Generalizes Poincaré-Lefschetz-style dualities from 1-parameter to multi-parameter persistence; concrete algorithm to compute free presentations of (co)homology of multiparameter Rips filtrations.
- Reality slot: optional `topology.MultiPersistence` (slot-351 paper 16's MMA was the entry point; this is the dual viewpoint).
- Pure-Go feasibility: ~1500 LOC if persistent homology is already present; otherwise much more.
- Recommend track? **Y partial.** Implement single-parameter persistence first (slot 351); use this to motivate the multi-parameter API design but defer the implementation.

### 7. Goyal et al. 2026 — Deformations, Derived Categories, and Multiparameter Persistence (arxiv:2604.10361)
- Result: Theoretical framework reframing multiparameter persistence as a deformation-theoretic / derived-categorical object; clarifies wild representation type and provides invariants stable under the framework.
- Reality slot: theory-only reference for `topology.MultiPersistence` design.
- Pure-Go feasibility: N/A (theoretical).
- Recommend track? **N implementation, Y document.**

### 8. Hou et al. 2026 — A U-match Algorithm for Persistent Relative Homology (arxiv:2602.03163)
- Result: Two-step matrix reduction giving persistent *relative* homology with worst-case complexity matching ordinary persistence; relative homology lifts the inference power of TDA when subspaces matter.
- Reality slot: `topology.PersistentRelativeHomology` once basic PH lands.
- Pure-Go feasibility: +200 LOC over ordinary boundary-matrix reduction.
- Recommend track? **Y partial.** Cheap addition once PH is in.

### 9. RCLA — Denoising Data Reduction for TDA (arxiv:2603.29248, Mar 2026)
- Result: Refined Characteristic Lattice Algorithm; grid-based denoising + reduction with stability bound (bottleneck distance) under homogeneous Poisson noise.
- Reality slot: `topology.RCLA` preprocessor for noisy point clouds.
- Pure-Go feasibility: ~400 LOC; just spatial hashing + threshold.
- Recommend track? **Y.** Pairs naturally with PH; small surface area.

### 10. Sun et al. 2026 — Persistent Homology as a Stable Chaos Measure (arxiv:2601.10900)
- Result: 0-dim PH-based chaos measure; provably more stable than maximal Lyapunov exponent under small state-space perturbations. Companion arxiv:2603.24675 ("Beyond largest Lyapunov") shows Shannon entropy diagnostics in Hénon-Heiles / N-body.
- Reality slot: `chaos.PersistentChaos` (extends existing Lyapunov module).
- Pure-Go feasibility: ~250 LOC if PH is already implemented.
- Recommend track? **Y partial.** Good cross-package value (uses PH primitive twice).

### 11. Ahmadi et al. 2026 — Geometric Perspective of Kalman Filters via Affine Connections (arxiv:2506.01086, with v2 in 2026)
- Result: Generalized Kalman framework on manifolds with affine connections; unifies Lie-group and Riemannian Kalman variants under a single intrinsic-geometry construction.
- Reality slot: extension of `geometry.LieGroup` + a new `prob.ManifoldKalman` (reality has no Kalman filter).
- Pure-Go feasibility: ~500 LOC if SE(3) / SO(3) primitives already exist (they do).
- Recommend track? **Y, prefer** if Kalman is added at all. Unifies the API design upfront.

### 12. NANO — Natural Gradient Gaussian Approximation Filter (arxiv:2605.02306, May 2026)
- Result: Bayesian filter doing natural-gradient descent on the statistical manifold of Gaussians; closes the long-running gap between Sigma-point / EKF and information-geometric filters.
- Reality slot: `prob.NaturalGradientFilter`; pairs with information geometry (`infogeo`).
- Pure-Go feasibility: ~350 LOC; needs Cholesky + Fisher-information of Gaussian (closed form).
- Recommend track? **Y.** Short, principled, lands in two reality packages at once.

### 13. RandRAND — Preconditioning via Randomized Range Deflation (arxiv:2509.19747, late 2025; with continued momentum into 2026)
- Result: Deflates the spectrum of a large linear system by orthogonal projection onto random subspaces, *without* eigenpair or low-rank computation; preconditioner cost is essentially Gaussian sketch + thin QR.
- Reality slot: `linalg.RandRANDPreconditioner` for CG / GMRES.
- Pure-Go feasibility: ~250 LOC; reality already has SRHT + thin QR.
- Recommend track? **Y, prefer** if a Krylov solver tier is added (slot 351 papers 2/3).

### 14. Conformalized Percentile Interval (arxiv:2605.03233, May 2026)
- Result: Tighter, conditionally-valid finite-sample CP intervals under heteroskedasticity / skewness / estimation error; outperforms CQR family on standard benchmarks with no exchangeability cost.
- Reality slot: extends `prob.Conformal` (with slot-351 ECI / multivariate-CP).
- Pure-Go feasibility: ~150 LOC; just a percentile-pair score function.
- Recommend track? **Y.** Trivial code addition that closes a known weakness.

### 15. ConformaDecompose — Calibration Localization (arxiv:2604.27149, Apr 2026)
- Result: Decomposes prediction-set width into per-feature calibration error contributions; turns conformal sets into interpretable diagnostics.
- Reality slot: `prob.ConformalDecompose` diagnostic helper.
- Pure-Go feasibility: ~200 LOC; sorting + leave-one-out residual ratios.
- Recommend track? **Y.** Diagnostic value with no math controversy.

### 16. Hankel Random Digital Net QMC (arxiv:2604.24105, Apr 2026)
- Result: Constructive randomized digital net using a Hankel structure; simpler to instantiate than Joe-Kuo direction numbers and gives matching average-case discrepancy with stronger probabilistic worst-case guarantees.
- Reality slot: `prob.QuasiMC` (with slot-351 paper 14 Sobol+Owen).
- Pure-Go feasibility: ~250 LOC; just GF(2) Hankel construction + bit-reversed iteration.
- Recommend track? **Y.** Avoids shipping Joe-Kuo tables (which are large; license is permissive but tables bloat the repo).

### 17. Kelly et al. 2026 — Steady-State Distributed Kalman (arxiv:2603.20013)
- Result: CCG-based fixed-structure steady-state estimator for discrete-time LTV systems; deterministic guarantees on covariance evolution.
- Reality slot: minor — useful as a reference once `prob.Kalman` lands.
- Pure-Go feasibility: ~300 LOC.
- Recommend track? **N initially.** Add only after baseline KF/EKF.

### 18. QuaSARQ / Clifft — Stabilizer & Near-Clifford Simulators (arxiv:2603.14641 GPU, arxiv:2604.27058 near-Clifford CPU)
- Result: GPU-parallel tableau evolution at 180k qubits / depth 1000 (105× speedup); Clifft factors a near-Clifford state into offline Clifford frame + online Pauli frame + dynamic active-state vector.
- Reality slot: `quantum/stabilizer` (slot 351 paper 20 was Acharya 2025 Nature; this is the simulator side).
- Pure-Go feasibility: CPU stabilizer simulator ~1500 LOC; near-Clifford active-subspace decomposition ~400 LOC delta.
- Recommend track? **Y partial.** CPU stabilizer simulator first; near-Clifford is an add-on.

### 19. Foundations of Riemannian Geometry monograph (arxiv:2605.02279, May 2026 final version)
- Result: Final, self-contained derivation-oriented monograph for Riemannian gradients/Hessians/exp/retract on Stiefel, Grassmann, SPD, oblique manifolds, with implementation pseudocode.
- Reality slot: reference for `optim.Riemannian` and `geometry.LieGroup`.
- Pure-Go feasibility: monograph itself is reference; the kernel implementations are ~600 LOC across manifolds (per slot 351 paper 19).
- Recommend track? **Y, supersedes** the slot-351 reference (older 2025 revision). Use the May 2026 v1 in any future implementation.

### 20. Composite Wavelet Matrix-Based Transforms (arxiv:2603.02593, Mar 2026)
- Result: Composite wavelet basis with stronger energy concentration than DB / Haar / Symlets; sparser denoising under identical thresholding.
- Reality slot: `signal.WaveletComposite` (reality has only basic FFT — no wavelet at all).
- Pure-Go feasibility: ~500 LOC; standard lifting scheme + composite filter banks.
- Recommend track? **N for now.** First add classic DB/Haar wavelets; revisit composite once baseline exists.

### 21. arxiv:2602.09996 — Learning to Choose Branching Rules for Nonconvex MINLPs (Feb 2026)
- Result: Regression-based branching-rule selector for outer-approximation MINLP solvers.
- Reality slot: out of scope for reality core (the *algorithm* is ML; the underlying outer-approximation B&B is classical and could go in `optim`, but reality has no MINLP).
- Pure-Go feasibility: classical B&B + Frank-Wolfe convex relaxation ~1500 LOC; ML selector excluded.
- Recommend track? **N.** MINLP is too far from reality's scope.

## Aggregate themes (4 months of 2026)

- **Graph theory had a real 2026:** the 2025 sorting-barrier result is generating a wave of follow-on linear-time SSSP / spanner / decomposition papers. Worth a focused `graph` upgrade pass in Q3 2026.
- **Persistent homology has graduated, *finally*:** 4 separate January–April 2026 papers on PH (denoising, relative, multi-parameter dualities, derived categories). The window for adding `topology.PersistentHomology` is now.
- **Conformal prediction continues its small-paper steady accumulation:** 2026 adds ConformaDecompose, conformalized percentile intervals, CP-as-Bayesian-quadrature framings — all tiny code patches with high marginal value.
- **Manifold/Lie-group methods are spreading from optimization into estimation:** 2026 brings ManifoldKalman + NANO + the final Riemannian-geometry monograph. A unified `geometry.Riemannian` module composing with `prob` and `optim` is the cleanest design.
- **Sum-check prover engineering wins keep arriving:** eprint 2026/587 is the third sum-check prover-speedup paper in 12 months. The protocol itself is stable; ship a clean reference implementation and bolt the optimizations on later.
- **Isogeny PQC is consolidating but not yet shippable:** ∆-SQIsign + the Castryck-Decru-resistant survey suggest the isogeny family will stabilize by 2027. Stay on ML-DSA / ML-KEM for now.

## Supersession notes (vs slot 351's 2025 list)

- Slot 351 paper 19 (Riemannian foundations monograph) → use **arxiv:2605.02279 v1, May 2026** as the reference instead of the older 2025 revision; content is updated and final.
- Slot 351 paper 16 (MMA multipersistence) → augment with arxiv:2603.18224 (dualities) and arxiv:2604.10361 (derived categories) before any implementation pass; the *theory* has moved.
- Slot 351 paper 8/9/10 (folding-ZK / sumcheck / lookup) → adopt eprint 2026/587 prover speedups when the sumcheck primitive is implemented.
- Slot 351 paper 6/7 (multivariate CP, ECI) → add arxiv:2605.03233 (conformalized percentile interval) and arxiv:2604.27149 (ConformaDecompose) as cheap follow-ons.
- Slot 351 paper 24 (ECPP record) → no 2026 movement; recommendation unchanged (BPSW first, ECPP later).
- Slot 351 paper 4 (ML-DSA / ML-KEM) → no 2026 supersession; ∆-SQIsign and friends are *additional*, not replacement.
- Slot 351 paper 20 (QEC below threshold) → arxiv:2603.14641 (GPU stabilizer) and arxiv:2604.27058 (Clifft near-Clifford) are the simulator side of the same theme; track but defer.

## Honest disclosure
The 2026 corpus through May 9 is genuinely thin. Of the 21 entries above, perhaps 8 are clearly novel-in-2026 contributions (the SSSP/spanner cluster, sumcheck eprint 2026/587, ∆-SQIsign, the four PH papers, conformalized percentile interval, ConformaDecompose, Hankel digital net). The remainder are 2025 work that received final revisions in 2026 or routine extensions. Recommend re-running this query in October 2026 once NeurIPS / ICLR submissions land.

## Sources

- arxiv:2603.22948 — Linear-time SSSP for Euclidean graph classes
- arxiv:2604.08667 — Faster shortest-paths via the acyclic-connected tree
- arxiv:2504.17033 — Breaking the sorting barrier (the 2025 base result)
- arxiv:2603.17085 — New greedy spanners and applications
- arxiv:2605.00837 — FastSinkhorn (log-domain Sinkhorn OT)
- arxiv:2603.21554 — Sinkhorn for entropic vector quantile regression
- eprint 2026/587 — Speeding up sum-check proving
- eprint 2026/852 — ∆-SQIsign degree-challenge signature
- eprint 2026/446 — Survey of Castryck-Decru-resistant isogeny signatures
- eprint 2026/032 — Algebraic isogeny model
- arxiv:2603.18224 — Dualities in multiparameter persistence
- arxiv:2604.10361 — Deformations, derived categories and multiparameter persistence
- arxiv:2602.03163 — U-match for persistent relative homology
- arxiv:2603.29248 — RCLA denoising for TDA
- arxiv:2601.10900 — PH-based chaos measure
- arxiv:2603.24675 — Entropy-based chaos diagnostics beyond Lyapunov
- arxiv:2506.01086 — Geometric Kalman via affine connections (v2 2026)
- arxiv:2605.02306 — NANO natural-gradient Gaussian filter
- arxiv:2603.20013 — Steady-state distributed Kalman
- arxiv:2509.19747 — RandRAND randomized range deflation
- arxiv:2605.03233 — Conformalized percentile interval
- arxiv:2604.27149 — ConformaDecompose calibration localization
- arxiv:2512.17048 — CP as calibration standard for HEP
- arxiv:2604.24105 — Hankel random digital net QMC
- arxiv:2603.14641 — GPU stabilizer simulator (QuaSARQ)
- arxiv:2604.27058 — Clifft near-Clifford simulation
- arxiv:2604.12841 — Fast AI pre-decoders for surface codes
- arxiv:2605.04892 — FPGA neural-network surface-code decoder
- arxiv:2605.02279 v1 — Foundations of Riemannian geometry (final, May 2026)
- arxiv:2603.02593 — Composite wavelet matrix-based transforms
- arxiv:2602.09996 — Branching rules for nonconvex MINLPs
- arxiv:2605.06607 — AI CFD scientist (turbulence; out of scope for reality)
- arxiv:2603.09693 — Physics-informed neural operator for phase-field PDEs
- arxiv:2601.14517 — PINN/NO unifying view (track only)
- arxiv:2601.14725 — Differential privacy on affine manifolds
- arxiv:2601.02397 — Evolutionary Nash equilibrium for dynamic games
- arxiv:2603.07974 — ZK-ACE post-quantum blockchain (Poseidon usage)
- arxiv:2602.07633 — Flow-based conformal predictive distributions
- arxiv:2603.04764 — Conformal Bayes filter for MIMO channels

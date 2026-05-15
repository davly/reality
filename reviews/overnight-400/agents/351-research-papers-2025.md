# 351 — research-papers-2025 (high-impact 2025 papers reality should track)

## Headline
2025 produced concrete, numerically-foundational results in randomized NLA, lattice PQC, folding-based ZK, NUFFT, multi-parameter persistence, conformal inference, and quasi-Monte Carlo — all MIT-compatible and several at <500 LOC pure Go. Reality should adopt 8–12 of these and track another ~10.

## Top papers (sorted by impact × fit)

### 1. Palitta & Portaro 2025 — Row-aware Randomized SVD (arxiv:2408.04503, rev. Jul 2025)
- Result: modifies standard rSVD to explicitly construct row-space information, with new spectral-gap-aware error bounds; same FLOP cost, strictly better accuracy when σ_k/σ_{k+1} is moderate.
- Reality slot: `linalg.RandomizedSVD` (drop-in upgrade to existing rSVD).
- Pure-Go feasibility: ~150 LOC delta over standard rSVD; needs only Gaussian sketch + QR + Hermitian eig.
- Recommend track? **Y, prefer.** Strict Pareto improvement, no library dependency, simple tolerance bump in golden tests.

### 2. Schmid et al. 2025 — Randomized Krylov-Schur Eigensolver with Deflation (arxiv:2508.05400)
- Result: rKS combines sketch-orthogonalization with Schur reordering and deflation; faster than ARPACK-style on large sparse symmetric/nonsymmetric problems while preserving stability.
- Reality slot: new `linalg.KrylovSchurEig` — fills a gap (reality currently has only power iteration / dense QR).
- Pure-Go feasibility: ~600 LOC; needs Hessenberg reduction, Schur reorder (complex 2x2 swaps), sketching.
- Recommend track? **Y, prefer** for k≪n eigenproblems. Foundational primitive.

### 3. Tropp & Webber 2024 (rev. 2025) — Fast and Accurate Randomized Algorithms for Linear Systems and Eigenvalue Problems (SIMAX, see arxiv:2502.01888 family)
- Result: randomized GMRES/eig with sketch-and-solve plus iterative refinement; provably backward stable, 2–10x faster than classical Krylov on ill-conditioned systems.
- Reality slot: `linalg.SketchedGMRES`, `linalg.SketchedLeastSquares`.
- Pure-Go feasibility: ~400 LOC; needs SRHT or Gaussian sketch + classical GMRES core (already feasible).
- Recommend track? **Y, prefer** when n ≥ 10⁴.

### 4. NIST FIPS 204 ML-DSA / FIPS 203 ML-KEM standards (2024–2025 deployment via OpenSSL 3.5, eprint 2025/2025)
- Result: lattice-based digital signature (Module-LWE/MSIS) and KEM, finalized standards with stable parameter sets; OpenSSL 3.5 ships them April 2025.
- Reality slot: new `crypto/mldsa` and `crypto/mlkem` subpackages.
- Pure-Go feasibility: ~2500 LOC each; NTT over Z_q, SHAKE-256, rejection sampling. No external deps.
- Recommend track? **Y, prefer.** PQC is the durable reference primitive going forward; SHA-2/ECDSA stay but PQC must be present.

### 5. Bouchard et al. 2025 — Asymptotically exact variational flows via involutive MCMC kernels (arxiv:2506.02162)
- Result: tuning-free variational families with provable total-variation convergence, built from involutive MCMC kernels as iterated random function systems; matches or beats NUTS on posterior approximation and normalization constants.
- Reality slot: `prob.InvolutiveFlow` — combines with existing Bayesian inference.
- Pure-Go feasibility: ~500 LOC; just MH/HMC kernel composition + measure-preserving inverses; no autodiff strictly required for symmetric kernels.
- Recommend track? **Y, prefer** over MixFlows. Strong theory + clean code path.

### 6. Cresswell-Clay et al. 2025 — Multivariate Conformal Prediction via Conformalized Gaussian Scoring (arxiv:2507.20941)
- Result: conformalized level sets of an estimated multivariate Gaussian density yield finite-sample valid prediction *regions* with strong conditional coverage; handles missing outputs.
- Reality slot: `prob.ConformalRegion` (extends current CP for scalars).
- Pure-Go feasibility: ~250 LOC; needs Cholesky + chi-squared quantile + nonconformity score sort (all already in reality).
- Recommend track? **Y, prefer** for multivariate UQ. Very small surface area.

### 7. Xu et al. 2025 — Error-quantified Conformal Inference for Time Series (ICLR 2025, arxiv:2502.00818)
- Result: smooths quantile pinball loss, gives continuous adaptive feedback to miscoverage; SOTA on time-series conformal coverage with smaller sets.
- Reality slot: `prob.ECI` time-series CP module.
- Pure-Go feasibility: ~150 LOC; just an EMA + smoothed quantile.
- Recommend track? **Y, prefer** over ACI/AgACI for streaming.

### 8. Folding-Based ZK Survey (Sci. of Comp. Prog. 2025, see arxiv:2505.20136 and Sonobe modules) + HyperPlonk acceleration (eprint 2025/620)
- Result: Nova/SuperNova/HyperNova folding unified; constraint models (R1CS / CCS / Plonkish) classified; HyperPlonk speed records.
- Reality slot: new `crypto/zk` (folding primitives + R1CS evaluator + sumcheck) — a foundational tier reality currently lacks.
- Pure-Go feasibility: Nova-lite ~3000 LOC; needs MSM, Pedersen commits, FFT over BN254 scalar field, sumcheck.
- Recommend track? **Y partial.** Implement sumcheck + Pedersen + R1CS evaluator first (~1200 LOC); full Nova later.

### 9. Mohamed 2025 — On Univariate Sumcheck (arxiv:2505.00554) + HybridPlonk Sub-Logarithmic SNARKs (eprint 2025/908)
- Result: clarifies univariate sumcheck soundness; HybridPlonk achieves O(n) prover with sub-logarithmic verifier.
- Reality slot: same `crypto/zk` tier.
- Pure-Go feasibility: univariate sumcheck ~300 LOC over BN254/BLS12-381 scalar field.
- Recommend track? **Y.** Sumcheck is the universal substrate; even without full SNARKs the primitive is reusable.

### 10. SoK: Lookup Table Arguments (eprint 2025/1876) + Logup
- Result: systematizes multiset / accumulator / Logup / matrix-vector lookup arguments; recommends Logup + cq variants for prover efficiency.
- Reality slot: `crypto/zk/lookup`.
- Pure-Go feasibility: Logup ~400 LOC over a prime field with KZG (or transparent variant ~250 LOC).
- Recommend track? **Y.** Composes with sumcheck above; small marginal cost.

### 11. Wang et al. 2025 — Solving Low-Rank SDPs via Manifold Optimization, ManiSDP (arxiv:2303.01722, rev. Apr 2025)
- Result: Burer-Monteiro + augmented Lagrangian on low-rank manifold; orders of magnitude faster than MOSEK/SDPNAL+ on low-rank SDPs.
- Reality slot: `optim.SDPLowRank` — reality has no SDP solver yet.
- Pure-Go feasibility: ~700 LOC; needs Riemannian gradient on the factor manifold + AL outer loop.
- Recommend track? **Y, prefer** if SDP is added at all (low-rank case is the practical one).

### 12. Hauswirth et al. 2025 — Projections onto Spectral Matrix Cones (arxiv:2511.01089)
- Result: closed-form / fast projections onto sum-of-k-largest-eigenvalues, nuclear, operator norm cones; integrated into SCS for ~10x speedup.
- Reality slot: `optim.ProjectSpectralCone` (composes with SCS-style first-order solver).
- Pure-Go feasibility: ~400 LOC; needs symmetric eig (have it) + sorted-eigenvalue projection.
- Recommend track? **Y.** Reusable beyond SDP — appears in robust PCA, low-rank recovery.

### 13. Magland et al. 2024–2025 — fftvis / FINUFFT Type-3 NUFFT (arxiv:2506.02130; FINUFFT v2.4)
- Result: Type-3 NUFFT (nonuniform → nonuniform) reduces baseline-grid simulators by 2 orders of magnitude; the underlying spread-and-FFT kernel is the canonical reference.
- Reality slot: `signal.NUFFT` (Types 1/2/3) — currently reality has only uniform FFT.
- Pure-Go feasibility: ~800 LOC; ES kernel spreader + existing FFT + zero-padding; no GPU needed for canonical version.
- Recommend track? **Y, prefer.** Foundational primitive for MRI, radio astronomy, NMR, and unevenly-sampled time series.

### 14. Lehec et al. 2025 — Neural Low-Discrepancy Sequences (arxiv:2510.03745) + QMC tutorial (arxiv:2502.03644)
- Result: tutorial consolidates Sobol/Halton/digital nets and randomization (scrambled nets, RQMC); neural sequences match Sobol+Owen scrambling on irregular domains.
- Reality slot: `prob.QuasiMC` — Sobol + Owen scrambling + Faure + Halton.
- Pure-Go feasibility: classical Sobol + Owen ~400 LOC; direction numbers from Joe-Kuo are public-domain tables.
- Recommend track? **Y, prefer** classical Sobol+Owen (skip neural variant — needs ML).

### 15. Ren et al. 2025 — MaxTDA: Robust Statistical Inference for Maximal Persistence (arxiv:2504.03897)
- Result: kernel-density + level-set rejection sampling produces persistence diagrams whose top features are not systematically deflated; restores statistical inference at the maximum.
- Reality slot: `topology.PersistentHomology` (new package) or under existing `geometry`.
- Pure-Go feasibility: persistence on Vietoris-Rips ~1500 LOC (boundary matrix reduction); MaxTDA wrapper +200 LOC.
- Recommend track? **Y partial.** Add persistent homology first; wrap MaxTDA later.

### 16. Loiseau et al. 2025 — Multi-parameter Module Approximation MMA (J. Appl. Comp. Topology 2025; doi:10.1007/s41468-025-00222-y)
- Result: efficient approximation algorithm with guarantees for 2-parameter persistence modules via matching functions; first practical multipersistence pipeline.
- Reality slot: `topology.MultiPersistence`.
- Pure-Go feasibility: ~2000 LOC; needs Gröbner-style reductions and interval-decomposition heuristic.
- Recommend track? **Track only**, defer implementation. Field still moving.

### 17. Verma et al. 2025 — CFO: Continuous-Time PDE Dynamics via Flow-Matched Neural Operators (arxiv:2512.05297)
- Result: flow matching for the RHS of a PDE avoids backprop through ODE solvers; up to 87% error reduction with 25% training data.
- Reality slot: out-of-scope for reality core (needs neural net), but the underlying *flow matching* objective is pure math.
- Pure-Go feasibility: flow-matching loss + RK45 ~500 LOC, but the operator is ML; reality should not embed neural nets.
- Recommend track? **N** core; **Y** as reference for `aicore`.

### 18. Kelly et al. 2025 — Adaptive SDE Solvers with No Lévy Area Bias (review arxiv:2508.11004 + SPaRK)
- Result: clarifies which step-size controllers are admissible for SDEs; introduces SPaRK splitting-path Runge-Kutta with adaptive step size and provable strong order.
- Reality slot: extend `chaos.SolveSDE` (currently fixed-step Euler-Maruyama / Milstein).
- Pure-Go feasibility: ~300 LOC; embedded Runge-Kutta Maruyama with PI step controller.
- Recommend track? **Y, prefer** for any user-facing SDE work.

### 19. Foundations of Riemannian Geometry for Riemannian Optimization (arxiv:2605.02279, rev. May 2025) + Momentum on Lie Groups (arxiv:2404.09363, rev. Jul 2025)
- Result: implementation-oriented monograph for Riemannian gradient/Hessian/exp/retract; companion gives intrinsic momentum methods on SO(n)/SE(3).
- Reality slot: `optim.Riemannian` and `geometry.LieGroup` extensions; reality has quaternions + SE(3) but not optimization on them.
- Pure-Go feasibility: Stiefel/Grassmann/SPD retractions ~600 LOC; momentum on SO(3) ~150 LOC over existing quaternions.
- Recommend track? **Y.** Plumb into existing geometry package — no new dependencies.

### 20. Acharya et al. 2025 — Quantum Error Correction Below the Surface Code Threshold (Nature 2025; arxiv:2408.13687, rev. 2025)
- Result: Google distance-7 surface code with logical error 0.143%/cycle, suppression factor 2.14× per +2 distance. The reference simulation toolkit (Stim + PyMatching) is the de facto QEC primitive.
- Reality slot: Out of scope for `reality` core, but *Pauli stabilizer simulation* (Gottesman-Knill) is pure linear algebra over GF(2) and would belong in a future `quantum/stabilizer` subpackage.
- Pure-Go feasibility: Stim-style stabilizer simulator ~1500 LOC; MWPM decoder ~600 LOC over existing graph package.
- Recommend track? **Y** as `quantum` package addition; ties into `graph` (matching).

### 21. Zhang et al. 2025 — Random Walk Neural Networks revisit (arxiv:2407.01214 ICLR 2025)
- Result: characterizes which random-walk-on-graph statistics give universal graph approximation; gives algorithm for sampling walks for graph-level tasks.
- Reality slot: `graph.RandomWalk` (random walk kernels, hitting times, commute distance).
- Pure-Go feasibility: ~250 LOC; reality already has graph + sparse matrix; just Markov chain stationary + first-passage.
- Recommend track? **Y partial.** Implement classical hitting/commute time; the GNN angle is ML.

### 22. Differentiating through SDEs: A Primer (arxiv:2601.08594, rev. 2025) + Common AD Interface (arxiv:2505.05542)
- Result: clear primer on adjoint vs. forward sensitivities through Itô/Stratonovich SDEs; companion proposes interface design that decouples AD engine from numerical code.
- Reality slot: `calculus.AutoDiff` (forward+reverse) — reality currently has finite differences only.
- Pure-Go feasibility: forward-mode dual numbers ~200 LOC; reverse-mode tape ~600 LOC. No allocations in hot path achievable.
- Recommend track? **Y, prefer** forward-mode. Reverse-mode is design-heavy; defer.

### 23. Conformal Prediction = Bayes? (arxiv:2512.23308)
- Result: formal separation results — conformal cannot in general replace Bayesian predictive inference; gives precise conditions where they coincide.
- Reality slot: documentation / theory pin in `prob.Conformal` — informs API design (separate calibrated frequentist and Bayesian paths).
- Pure-Go feasibility: N/A (theoretical).
- Recommend track? **Y** as design reference; no implementation.

### 24. Underwood (May 2025) — ECPP record R(109297) via Enge fastECPP
- Result: practical ECPP performance milestone; affirms Atkin-Morain Õ(L⁴) is still the right algorithm for arbitrary primality proofs.
- Reality slot: `crypto.PrimalityCertificate` (ECPP) — reality has Miller-Rabin only.
- Pure-Go feasibility: full ECPP ~3000 LOC (CM method, Hilbert class polynomial, point counting). Heavy.
- Recommend track? **Y partial.** APR-CL or BPSW first; ECPP is correct goal but high LOC.

### 25. SDE Matching: Simulation-Free Latent SDE Training (arxiv:2502.02472)
- Result: matches latent SDE drift against a target without simulating; competitive accuracy at far lower cost.
- Reality slot: out of scope for reality core (ML); list as reference.
- Pure-Go feasibility: N/A.
- Recommend track? **N.**

## Aggregate themes

- **Randomized numerical linear algebra is the bedrock** (papers 1, 2, 3): row-aware rSVD, randomized Krylov-Schur, sketched GMRES — reality should standardize on Gaussian/SRHT sketching as a primitive in `linalg.Sketch` and rebuild rSVD/eig/least-squares around it.
- **Conformal/Bayesian fusion** (papers 6, 7, 23): multivariate CP, time-series ECI, and the CP=Bayes? separations make a coherent UQ chapter for `prob`. Add together, not piecemeal.
- **Folding-based ZK is finally ready** (papers 8, 9, 10): Nova/HyperPlonk/Logup unified by sumcheck. Reality should add a `crypto/zk` foundation (sumcheck + Pedersen + lookup) without committing to a specific frontend.
- **PQC is non-negotiable** (paper 4): ML-DSA + ML-KEM as `crypto/mldsa`, `crypto/mlkem`. Standards are stable, code is mechanical.
- **Persistence/topology has graduated** (papers 15, 16): MaxTDA + multipersistence approximation give a 2025 baseline for a `topology` package.
- **PDE/SDE numerics are bifurcating into classical+ML** (papers 17, 18, 22): reality should track the *classical* SDE adaptive-step + flow-matching primitive; leave the neural operators to `aicore`.
- **Riemannian/Lie group optimization is mature enough to ship** (paper 19): retractions for SO(n), Stiefel, SPD, plus intrinsic momentum on SO(3)/SE(3) compose with existing `geometry`.
- **Special primitives reality is missing** (papers 13, 14, 24): NUFFT, scrambled-net QMC, ECPP — each is a 200–800 LOC pure-math addition with broad downstream value.
- **Deferred (track but do not implement)**: full Nova/HyperNova frontend, neural-PDE operators, surface-code experimental decoders, neural low-discrepancy generators, multi-parameter persistence module decomposition.

## Sources

- arxiv:2408.04503 — Row-aware Randomized SVD (Palitta, Portaro)
- arxiv:2508.05400 — Randomized Krylov-Schur eigensolver
- arxiv:2502.01888 — Randomized block-Krylov for low-rank matrix functions
- arxiv:2505.20602 — Connecting randomized iterative methods with Krylov subspaces
- arxiv:2508.20269 — Randomized Krylov methods for inverse problems
- arxiv:2506.06882 — Randomized SVD in infinite dimensions
- arxiv:2505.23582 — S^TS-SVD via sketching
- NIST FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA); eprint 2025/2025 (ML-DSA migration)
- arxiv:2506.02162 — Asymptotically exact variational flows via involutive MCMC kernels
- arxiv:2507.20941 — Multivariate CP via Conformalized Gaussian Scoring
- arxiv:2502.00818 — Error-quantified Conformal Inference for Time Series (ICLR 2025)
- arxiv:2512.23308 — Conformal Prediction = Bayes?
- arxiv:2505.20136 — Folding-based ZK survey (Sci. Comp. Prog. 2025)
- eprint 2025/620 — zkSpeed / HyperPlonk acceleration
- eprint 2025/908 — HybridPlonk: Sub-logarithmic linear-time SNARKs
- eprint 2025/1876 — SoK: Lookup Table Arguments
- arxiv:2505.00554 — On Univariate Sumcheck
- arxiv:2303.01722 (rev. Apr 2025) — ManiSDP, low-rank SDPs via manifold optimization
- arxiv:2511.01089 — Projections onto Spectral Matrix Cones
- arxiv:2506.02130 — fftvis (Type-3 NUFFT) ; FINUFFT toolkit
- arxiv:2510.03745 — Neural Low-Discrepancy Sequences
- arxiv:2502.03644 — Quasi-Monte Carlo: What, Why, and How?
- arxiv:2504.03897 — MaxTDA
- doi:10.1007/s41468-025-00222-y — Multi-parameter Module Approximation (MMA)
- arxiv:2512.05297 — CFO: flow-matched neural operators for PDE dynamics
- arxiv:2508.11004 — Modern Stochastic Modeling review (SDE/SPDE numerics, SPaRK)
- arxiv:2605.02279 (rev. May 2025) — Foundations of Riemannian Geometry for Optimization
- arxiv:2404.09363 (rev. Jul 2025) — Momentum on Lie Groups
- arxiv:2408.13687 (Nature 2025) — QEC below surface-code threshold
- arxiv:2407.01214 — Random Walk Neural Networks (ICLR 2025)
- arxiv:2601.08594 — Differentiating through SDEs (primer)
- arxiv:2505.05542 — A Common Interface for Automatic Differentiation
- arxiv:2506.00796 — Higher-Order AD via Symbolic Differential Algebra
- ECPP record R(109297), Underwood / Enge fastECPP, May 2025
- arxiv:2502.02472 — SDE Matching (latent SDE training)

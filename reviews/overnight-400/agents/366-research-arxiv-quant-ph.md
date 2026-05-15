# 366 — research-arxiv-quant-ph (quant-ph results applicable to classical libs)

## Headline
arXiv quant-ph yields ~10 classical-math primitives reality could harvest: tensor-train/MPS arithmetic, GF(2) symplectic/Clifford enumeration, ℓ2-norm sampling sketches (Tang dequantization), QFT-as-FFT confirmation, Pauli-string exponential closed forms, BP/union-find graph decoders, and Szegedy walk operators on stochastic matrices.

## Top results (sorted by classical-applicability)

### 1. Tang 2019 / Chia et al. 2020 — ℓ2-norm sampling sketch (dequantization)
arXiv:1807.04271, 1910.06151, 2304.04932. Tang's recommendation-system dequantization established a classical algorithmic framework: given an m×n matrix in a length-squared sampling data structure, sample from a rank-k approximation in O(poly(k) log(mn)). Chia et al. generalized to a full Singular Value Transformation framework — classical SVT in time independent of dimension, recovering dequantizations of PCA, SVM, low-rank regression, semidefinite programs. **Classical primitive**: a `prob.LengthSquaredSampler` over rows/cols with O(log n) preprocessing and O(log n) query, plus a "matrix sketch" struct for low-rank approximate SVD via importance sampling. Pure linear algebra. Already deployed in randomized-NLA (Drineas, Mahoney). Robust dequantization (2304.04932) added approximate-sampling tolerance — practical when exact ℓ2 access is impossible.

### 2. Aaronson-Gottesman 2004 — stabilizer simulation in tableau form
arXiv:quant-ph/0406196. The CHP algorithm represents an n-qubit stabilizer state as a 2n×(2n+1) GF(2) matrix (binary symplectic + phase bit) and updates it under Clifford gates in O(n) per gate, full simulation in O(n² gates·n). **Classical primitive**: GF(2) matrix arithmetic with rowsum, symplectic inner product over (ℤ/2)^{2n}, Gaussian elimination over GF(2). The "tableau" is just a binary-symplectic linear-algebra package; it has applications outside QC (LDPC decoder syndrome compression, code-based crypto). reality currently has no GF(2) linear algebra. Add `linalg/gf2` or `crypto/gf2` with: rowsum, rank, RREF, symplectic Gram, transvection.

### 3. QFT ≡ classical DFT on amplitude vector (modulo bit-reversal)
Wikipedia + Pennylane Demo "QFT and groups". QFT_N |x⟩ = (1/√N) Σ_y ω^{xy} |y⟩, ω = e^{2πi/N}. Acting on amplitude vectors this **is** the DFT matrix (sign convention aside). The QFT circuit's recursive structure is the Cooley-Tukey FFT butterfly. **Reality already has signal/FFT**. Action: add a doc note in `signal/fft.go` explaining the QFT correspondence; consider exposing `fft.Bitreverse(n int) []int` and `fft.TwiddleTable(n int) []complex128` as first-class primitives — these are the building blocks both of FFT and (formally) of the QFT circuit decomposition.

### 4. White DMRG / MPS canonical form — block-tridiagonal eigensolver
arXiv:2503.08626, 2506.12629 (Software Landscape for DMRG, 2025). DMRG is a variational sweep algorithm: at each step solve a small effective Hamiltonian eigenvalue problem H_eff v = E v on a tensor product space, then SVD-truncate to bond dimension χ. Stripped of its quantum-chemistry interpretation, DMRG is a **block-Lanczos eigensolver with periodic SVD truncation on a 1D tensor train**. The math is: SVD with rank truncation, Krylov eigensolve, environment contraction = a chain of GEMMs. Classical applications: high-dimensional PDE solvers (Oseledets tensor-train), Markov-chain stationary distributions in tensor-train form, recommender systems on tensor data. **Primitive**: a `linalg/tt` (tensor-train) package — TT-SVD, TT-rounding, TT-MatVec, TT inner product. All standard BLAS3.

### 5. Clifford group enumeration via binary symplectic
arXiv:1306.3643 ("How to efficiently select an arbitrary Clifford group element"). The Clifford group on n qubits modulo Pauli is Sp(2n,2). |C_n| = 2^{n²+2n} · Π(4^i - 1). The enumeration algorithm produces the i-th element in O(n³) and indexes elements in O(n³). **Classical primitive**: enumeration of binary symplectic matrices Sp(2n,2). This is pure finite-group combinatorics — combinatorics package candidate. Applications: code-based crypto (McEliece variants), randomized benchmarking for sensors (not just qubits), Reed-Muller code construction.

### 6. Surface code decoders — union-find on syndrome graphs
arXiv:2307.14989, q-2024-10-10-1498, q-2021-12-02-595. Delfosse-Nickerson "almost-linear time" union-find decoder on the toric code. The decoder is **graph union-find with cluster-growth weights** plus minimum-weight perfect matching (Edmonds blossom) on dual edges. **Classical primitives reality could add**: (a) weighted union-find with rank/path-compression and cluster-weight tracking; (b) minimum-weight perfect matching on general graphs (Edmonds blossom). Both are textbook graph algorithms. reality's `graph/` has Dijkstra/A*/topo but no MWPM. MWPM is broadly useful (assignment, matching, scheduling).

### 7. Belief propagation on Tanner graphs (LDPC decoding)
arXiv:2312.10950, ldpc/quantumgizmos library, npj Quantum Information 2025 "ML message-passing for QLDPC". BP is the sum-product algorithm on a bipartite (variable, check) factor graph: O(E) per iteration, O(iter·E) total. Quantum-LDPC research has driven BP enhancements (BP+OSD = ordered statistics decoding fallback, BP+LSD = localized statistics decoder, guided decimation). **Classical primitives**: `prob/bp` — sum-product on factor graphs, max-product (MAP), Tanner-graph LDPC decode, ordered-statistics post-processing. Useful for general probabilistic inference, error correction, Bayesian networks.

### 8. Pauli exponential closed form: e^{iθP} = cos(θ)I + i sin(θ)P
arXiv:2305.04807, Wikipedia "Matrix exponential". Any Pauli string P satisfies P² = I, so its exponential is the trivial cos+i·sin form — no Padé, no scaling-and-squaring needed. This generalizes: any matrix M with M² = ±I has a closed-form exponential. **Classical primitive**: `linalg.ExpInvolution(M, theta)` for involutory matrices (M² = I). Useful in geometry (reflections, Householder), Lorentz boosts, Lie algebra of so(3). reality's `linalg` should expose `ExpSkew(omega)` (Rodrigues) and `ExpInvolution`.

### 9. Szegedy quantum walk → classical Markov chain operator
arXiv:2307.14314 (SQUWALS), 1611.02238. Szegedy's walk quantizes a stochastic matrix P by a unitary acting on edges. The classical analog is the **discriminant matrix** D(P) with D_{ij} = √(p_{ij} p_{ji}); the walk's spectral gap is governed by 1−σ²(D), where σ is the second singular value. **Classical primitive**: `prob.MarkovDiscriminant(P)` returning D, plus `prob.SpectralGap(P)`. Useful for MCMC mixing time bounds, PageRank acceleration, hitting-time computation.

### 10. Magic-state distillation ↔ classical weight enumerators
arXiv:2501.10163, royalsocietypublishing rspa 2020 (ternary Golay), Nature Physics 2025 "Constant-overhead". Distillation routines are governed by **classical code weight enumerators** (MacWilliams identity, simple/biweight enumerators over GF(2), GF(4)). **Classical primitives**: `crypto/codes` — MacWilliams transform on weight enumerator polynomials, biweight enumerator computation, code-distance bounds (Singleton, Hamming, Plotkin, Griesmer). reality has no error-correcting-code package; this is foundational classical-coding-theory math, used widely in CD/DVD, deep-space telemetry (Reed-Solomon), and 5G (LDPC).

### 11. Tensor-train / MPS rounding (TT-SVD + bond-truncation)
arXiv:2501.18263 (TN methods for strongly correlated systems), Boulder-school 2025 lectures. The TT-rounding operation: given a tensor-train representation with bond dimensions r_i, find the best representation with r_i' ≤ r_i in Frobenius norm. Algorithm: left-to-right QR sweep, right-to-left SVD-truncate sweep. O(n d r³) flops, n cores, d physical dim, r max bond. **Classical primitive**: `linalg/tt.Round(tt, eps)`. Useful for high-dim function compression, Boltzmann-machine partition functions, image compression in scientific computing.

### 12. Binary symplectic transvections (Clifford generators)
arXiv:1803.06987, 1907.00310 (Logical Clifford Synthesis). A symplectic transvection T_h(v) = v + ⟨v,h⟩·h on (GF(2))^{2n} generates Sp(2n,2). Every symplectic matrix decomposes into ≤ 2n transvections (Salam-style). **Classical primitive**: GF(2) symplectic transvection composition + decomposition. Group-theoretic foundation; small but elegant. Combine with primitive #5.

## Reality slot recommendations

1. **`linalg/gf2`** (NEW) — GF(2) matrix arithmetic, RREF, rank, symplectic Gram, transvection. Justifies #2, #5, #10, #12. Foundational for code-based crypto, LDPC, Clifford enumeration.
2. **`linalg/tt`** (NEW) — Tensor-train SVD, rounding, MatVec, contractions. Justifies #4, #11. Useful well beyond QC (Oseledets PDE, Markov chains in TT format).
3. **`prob.LengthSquaredSampler`** + low-rank sketch — Tang/Chia framework. Justifies #1. Pure randomized-NLA.
4. **`prob/bp`** — sum-product, max-product, factor graphs, ordered statistics. Justifies #7. Bayesian-inference workhorse.
5. **`graph.MinWeightPerfectMatching`** (Edmonds blossom) and `graph.UnionFindWeighted`. Justifies #6.
6. **`prob.MarkovDiscriminant` + `prob.MarkovSpectralGap`** — Szegedy classical analog. Justifies #9.
7. **`linalg.ExpInvolution(M, theta)`** — closed-form matrix exponential when M² = I. Trivial addition to existing linalg. Justifies #8.
8. **`crypto/codes`** (NEW) — MacWilliams transform, biweight enumerators, Singleton/Hamming/Griesmer bounds. Justifies #10.
9. **`signal/fft` doc note** — explicitly link FFT and QFT (formal equivalence). Expose `Bitreverse`, `TwiddleTable`. Justifies #3.
10. **`combinatorics.Sp2n2Enumerate(n, i)`** — i-th binary symplectic matrix in O(n³). Justifies #5, #12.

Priority ranking for v0.11/v0.12: gf2 > tt > bp > MWPM > LengthSquaredSampler > codes > rest.

## Sources

- arXiv:1807.04271 — Tang, "Quantum-inspired classical algorithm for recommendation systems" (https://arxiv.org/abs/1807.04271)
- arXiv:1910.06151 — Chia/Gilyén/Li/Lin/Tang/Wang, "Sampling-based sublinear low-rank matrix arithmetic" (https://arxiv.org/abs/1910.06151)
- arXiv:2304.04932 — "Robust Dequantization of QSVT" (https://arxiv.org/html/2304.04932)
- arXiv:quant-ph/0406196 — Aaronson-Gottesman, "Improved Simulation of Stabilizer Circuits" (https://arxiv.org/abs/quant-ph/0406196)
- arXiv:2503.08626 — "Tensor networks for quantum computing" review (https://arxiv.org/html/2503.08626v1)
- arXiv:2506.12629 — "The Software Landscape for DMRG" (2025) (https://arxiv.org/pdf/2506.12629)
- arXiv:2501.18263 — "Tensor network state methods and quantum information theory" (2025) (https://arxiv.org/html/2501.18263v1)
- arXiv:2307.14989 — "Review on the decoding algorithms for surface codes" (https://arxiv.org/html/2307.14989v4)
- Quantum Journal q-2024-10-10-1498 — "Decoding algorithms for surface codes" (https://quantum-journal.org/papers/q-2024-10-10-1498/)
- Quantum Journal q-2021-12-02-595 — Delfosse-Nickerson "Almost-linear time decoding" (https://quantum-journal.org/papers/q-2021-12-02-595/)
- arXiv:2312.10950 — "BP Decoding of Quantum LDPC with Guided Decimation" (https://arxiv.org/abs/2312.10950)
- npj Quantum Information 2025 — "ML message-passing for scalable QLDPC decoding" (https://www.nature.com/articles/s41534-025-01033-w)
- arXiv:2305.04807 — "Decomposition Algorithm of an Arbitrary Pauli Exponential" (https://arxiv.org/pdf/2305.04807)
- arXiv:2307.14314 — SQUWALS Szegedy walks simulator (https://arxiv.org/abs/2307.14314)
- arXiv:1611.02238 — "Equivalence of Szegedy's and Coined Quantum Walks" (https://arxiv.org/abs/1611.02238)
- arXiv:2501.10163 — "Invariant Theory, Magic State Distillation, and Bounds on Classical Codes" (2025) (https://arxiv.org/html/2501.10163)
- Nature Physics 2025 — "Constant-overhead magic state distillation" (https://www.nature.com/articles/s41567-025-03026-0)
- royalsocietypublishing rspa 2020 — "Magic state distillation with the ternary Golay code"
- arXiv:1803.06987 / 1907.00310 — Rengaswamy et al., "Synthesis of Logical Clifford Operators via Symplectic Geometry"
- Wikipedia: Quantum Fourier transform, Gottesman-Knill theorem, Matrix exponential
- Pennylane "QFT and groups" demo (https://pennylane.ai/qml/demos/tutorial_qft_and_groups)
- ldpc by quantumgizmos (https://github.com/quantumgizmos/ldpc)

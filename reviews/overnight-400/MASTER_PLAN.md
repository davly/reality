# Reality — 400-Agent Overnight Review Plan

Generated 2026-05-06. Each line is a distinct review angle to be assigned to one agent.
Format: `NNN | slug | topic`

Agents read this file to find their assigned topic. Output goes to `agents/NNN-slug.md`.

## Block A — Per-package depth (150)

Five angles per package: (1) numerical accuracy & edge cases, (2) missing canonical algorithms, (3) state-of-the-art comparison via web research, (4) API/ergonomics, (5) performance / allocation hot-path audit.

001 | acoustics-numerics | acoustics: numerical accuracy & IEEE-754 edge case audit
002 | acoustics-missing  | acoustics: canonical algorithms missing (Schroeder integration, IR convolution reverb, HRTF, modal analysis, Westervelt, Kuznetsov)
003 | acoustics-sota     | acoustics: compare with libroom/pyroomacoustics, libsofa, openAcoustics — what's beyond what's there
004 | acoustics-api      | acoustics: API ergonomics, naming, units, return-types
005 | acoustics-perf     | acoustics: hot-path allocations, vectorization opportunities
006 | audio-numerics     | audio: numerical accuracy of onset/pitch/STFT, framing, windowing edge cases
007 | audio-missing      | audio: missing — CQT, mel-spectrogram, MFCC, BFCC, gammatone, beat tracking, source separation primitives
008 | audio-sota         | audio: compare with librosa, essentia, aubio, torchaudio
009 | audio-api          | audio: API ergonomics for streaming vs batch, dtype handling
010 | audio-perf         | audio: STFT/FFT throughput, ring buffers, SIMD opportunities
011 | autodiff-numerics  | autodiff: gradient accuracy, second-order tape correctness, complex-step diff
012 | autodiff-missing   | autodiff: missing — vjp/jvp/Jacobian/Hessian helpers, checkpointing, mixed-mode, source-to-source
013 | autodiff-sota      | autodiff: compare with JAX, Enzyme, Stalin∇, Tapenade, Zygote
014 | autodiff-api       | autodiff: ergonomics — `Var`/`Grad`/`Tape` naming, broadcasting, mutability
015 | autodiff-perf      | autodiff: tape memory, fusion, in-place ops
016 | calculus-numerics  | calculus: Simpson/RK4 truncation error analysis, IEEE-754 edge cases, root-finding convergence
017 | calculus-missing   | calculus: missing — adaptive Gauss-Kronrod, Romberg, exponentially-convergent integrators, Chebyshev quadrature, tanh-sinh, IMT
018 | calculus-sota      | calculus: compare with QUADPACK, Boost.Math.Quadrature, scipy.integrate, Mathematica's NIntegrate
019 | calculus-api       | calculus: function-handle ergonomics, error reporting, tolerance contracts
020 | calculus-perf      | calculus: vectorized integrand evaluation, allocation-free RK4 step
021 | changepoint-numerics | changepoint: detection-rate vs false-alarm calibration, edge cases
022 | changepoint-missing | changepoint: missing — Bayesian online (BOCPD), PELT, e-divisive, kernel CPD, BinSeg, WBS, NOT, GFL
023 | changepoint-sota   | changepoint: compare with ruptures, changepoint, bcp, mcp
024 | changepoint-api    | changepoint: streaming vs batch API, multi-variate handling
025 | changepoint-perf   | changepoint: O(n²) → O(n log n) opportunities, summed-area tricks
026 | chaos-numerics     | chaos: long-horizon ODE error growth, Lyapunov exponent estimator bias
027 | chaos-missing      | chaos: missing — Kuramoto-Sivashinsky, Henon, Rossler, Mackey-Glass, Duffing, Chua, Belousov-Zhabotinsky
028 | chaos-sota         | chaos: compare with DynamicalSystems.jl, JiTCDDE, ChaosBook resources
029 | chaos-api          | chaos: ODE callback ergonomics, event detection, dense output
030 | chaos-perf         | chaos: stiff vs non-stiff dispatch, BLAS in implicit solvers
031 | color-numerics     | color: gamut mapping, ΔE000 corner cases, white-point conversion accuracy
032 | color-missing      | color: missing — OKLab, OKLch, ICtCp, ITP, JzAzBz, CAM16-UCS, ipt-dolby
033 | color-sota         | color: compare with colour-science, color-rs, blender's OCIO
034 | color-api          | color: type safety for color-spaces, conversion fluency
035 | color-perf         | color: matrix conversion fusion, batch transforms
036 | combinatorics-numerics | combinatorics: integer overflow audits in Catalan/Stirling/Bell, exactness contracts
037 | combinatorics-missing | combinatorics: missing — Young tableaux, RSK correspondence, Ferrers diagrams, partition lattice, Möbius, Tutte poly, chromatic poly
038 | combinatorics-sota | combinatorics: compare with FLINT, Sage's combinat, Mathematica, OEIS-aware libs
039 | combinatorics-api  | combinatorics: big-integer interface, lazy generation
040 | combinatorics-perf | combinatorics: memoization, DP table layouts
041 | compression-numerics | compression: entropy estimation bias, Huffman optimality edge cases
042 | compression-missing | compression: missing — arithmetic coding, ANS/rANS, BWT, MTF, dictionary methods, Brotli/Zstd primitives, deltas for floats (FPC, Pcodec)
043 | compression-sota   | compression: compare with zstd, brotli, FPzip, ZFP, SZ, blosc
044 | compression-api    | compression: streaming codec interface, frame boundaries
045 | compression-perf   | compression: bit-IO efficiency, table lookups, SIMD
046 | constants-numerics | constants: CODATA 2022 vs 2018 deltas, unit conversion round-trip exactness
047 | constants-missing  | constants: missing constants and unit systems (Planck, atomic, geometrized)
048 | constants-sota     | constants: compare with NIST CODATA portal, scipy.constants, Boost.Units
049 | constants-api      | constants: dimensional analysis types vs raw float
050 | constants-perf     | constants: compile-time evaluation, inlining
051 | control-numerics   | control: PID windup, transfer-function pole-zero numerical stability, Bode at ω→0/∞
052 | control-missing    | control: missing — LQR, LQG/Kalman, MPC, H∞, μ-synthesis, IDA-PBC, sliding mode, gain scheduling
053 | control-sota       | control: compare with python-control, Slycot, do-mpc, casadi
054 | control-api        | control: state-space type, discretization API, frequency-domain ops
055 | control-perf       | control: matrix exponentials, Riccati solvers
056 | crypto-numerics    | crypto: modular-arithmetic correctness, constant-time guarantees, Miller-Rabin error bounds
057 | crypto-missing     | crypto: missing — EdDSA, BLS, Schnorr, Pedersen commitments, Shamir SS, VSS, oblivious transfer, garbled circuits, ECIES
058 | crypto-sota        | crypto: compare with libsodium, RustCrypto, Crypto++, dalek, Arkworks
059 | crypto-api         | crypto: hazmat surface, misuse resistance, key types
060 | crypto-perf        | crypto: Montgomery/Barrett reduction, NTT for big-int multiply, side-channel constant-time
061 | em-numerics        | em: field superposition cancellation errors, FDTD CFL stability
062 | em-missing         | em: missing — FDTD, MoM, FEM-EM, Smith chart, transmission line, antenna patterns, dipole radiation, waveguide modes
063 | em-sota            | em: compare with MEEP, openEMS, FEKO, COMSOL ACDC
064 | em-api             | em: vector field types, complex-valued fields, units
065 | em-perf            | em: stencil ops, sparse matrix assembly
066 | fluids-numerics    | fluids: Bernoulli ill-conditioning at low Δp, Reynolds-regime branching tests
067 | fluids-missing     | fluids: missing — Navier-Stokes, LBM, SPH, vorticity-streamfunction, Spalding, Colebrook iterations, Moody, Karman-Tsien
068 | fluids-sota        | fluids: compare with OpenFOAM primitives, lattice-Boltzmann libs, scipy.optimize.brentq for Colebrook
069 | fluids-api         | fluids: dimensional input validation, regime-aware dispatch
070 | fluids-perf        | fluids: pipe-network solvers, sparse Jacobians
071 | gametheory-numerics | gametheory: equilibrium computation tolerance, Lemke-Howson edge cases
072 | gametheory-missing | gametheory: missing — correlated equilibrium, ε-Nash, Stackelberg, mechanism design, VCG, auction primitives, fictitious play, CFR/MCCFR
073 | gametheory-sota    | gametheory: compare with Gambit, OpenSpiel, nashpy, Pluribus papers
074 | gametheory-api     | gametheory: payoff matrix vs extensive-form representation
075 | gametheory-perf    | gametheory: large game compression, abstraction
076 | geometry-numerics  | geometry: predicate robustness (Shewchuk-style), quaternion drift, SDF distance error
077 | geometry-missing   | geometry: missing — Delaunay, Voronoi, alpha shapes, RANSAC, ICP, Möller-Trumbore, BVH, k-d tree, R-tree, half-edge mesh
078 | geometry-sota      | geometry: compare with CGAL, GeometricTools, libigl, OpenMesh
079 | geometry-api       | geometry: point/vector/normal type distinction, immutable transforms
080 | geometry-perf      | geometry: spatial index queries, SIMD for batch transforms
081 | graph-numerics     | graph: weighted Dijkstra precision under negative-zero, Floyd-Warshall accumulating error
082 | graph-missing      | graph: missing — Johnson, Bellman-Ford, max-flow (Dinic, push-relabel), min-cut, matching (Hopcroft-Karp, blossom), Tarjan SCC, articulation points, k-cores, modularity, Louvain, Leiden, PageRank, HITS
083 | graph-sota         | graph: compare with NetworkX, igraph, graph-tool, Gephi, Boost.Graph
084 | graph-api          | graph: edge-list vs adjacency, directed/undirected types, weight semantics
085 | graph-perf         | graph: CSR layout, parallel BFS, GraphBLAS-style ops
086 | info-numerics      | info: log-probability stability, KL divergence with zeros, MI estimator bias
087 | info-missing       | info: missing — Rényi divergences, f-divergences, Tsallis, KSG MI estimator, MINE, conditional MI, transfer entropy, integrated information
088 | info-sota          | info: compare with JIDT, dit, npeet, mpmath
089 | info-api           | info: probability vs sample-based interfaces
090 | info-perf          | info: histogram binning, KSG nearest-neighbor speed
091 | infogeo-numerics   | infogeo: Fisher information matrix conditioning, geodesic ODE drift
092 | infogeo-missing    | infogeo: missing — α-connections, dual flat coordinates, e/m projections, mixture-family geodesics, natural gradient, Wasserstein gradient flows
093 | infogeo-sota       | infogeo: compare with Geomstats, Pymanopt, JAX-Cosmo
094 | infogeo-api        | infogeo: manifold types, exp/log map abstractions
095 | infogeo-perf       | infogeo: matrix-square-root caching, retraction shortcuts
096 | linalg-numerics    | linalg: pivot strategies, condition number reporting, iterative refinement
097 | linalg-missing     | linalg: missing — randomized SVD, Lanczos, Arnoldi, GMRES, BiCGStab, IDR(s), banded solvers, sparse Cholesky, AMD/RCM ordering, Schur, polar, generalized eig
098 | linalg-sota        | linalg: compare with LAPACK/Eigen/Armadillo/Blaze, MKL/OpenBLAS for kernel design
099 | linalg-api         | linalg: matrix view types, in-place vs copy semantics, broadcasting
100 | linalg-perf        | linalg: cache-blocked GEMM, register tiling, AVX-512/NEON
101 | optim-numerics     | optim: line-search robustness, BFGS positive-definiteness, simplex ill-conditioning
102 | optim-missing      | optim: missing — Adam/RMSprop, conjugate gradient, trust-region, IPOPT-style barriers, ADMM, proximal methods, Frank-Wolfe, CMA-ES, SPSA, NSGA-II, Bayesian optimization, hyperband, BOHB
103 | optim-sota         | optim: compare with NLopt, scipy.optimize, Optuna, ax, BoTorch, Optimization.jl
104 | optim-api          | optim: callback hooks, constraint specification, problem types
105 | optim-perf         | optim: gradient evaluation parallelism, batched objectives
106 | orbital-numerics   | orbital: Kepler-equation iteration convergence near e=1, perturbation accumulation
107 | orbital-missing    | orbital: missing — Lambert solver, n-body N≥3 propagators, J2/J3 zonal harmonics, SGP4, Cowell, Encke, regularization (KS, Sundman)
108 | orbital-sota       | orbital: compare with poliastro, Orekit, GMAT, STK
109 | orbital-api        | orbital: state vector vs orbital element, frame transforms (ICRF, ITRF)
110 | orbital-perf       | orbital: symplectic propagators, batched ephemeris
111 | physics-numerics   | physics: stress-tensor symmetry, thermodynamic equation-of-state stability
112 | physics-missing    | physics: missing — relativistic kinematics, special-relativity boosts, Lagrangian/Hamiltonian primitives, Noether currents, classical EM stress tensor
113 | physics-sota       | physics: compare with PyCav, Brian2 (neuro), Manim physics, Modelica
114 | physics-api        | physics: unit-aware types, vector/tensor abstractions
115 | physics-perf       | physics: tensor contractions, einsum-style
116 | prob-numerics      | prob: log-pdf/log-cdf stability, tail accuracy, regularized incomplete beta/gamma
117 | prob-missing       | prob: missing — Skellam, NIG, Levy, multivariate t, GIG, generalized hyperbolic, Tweedie, copulas (full family), Hawkes, Cox processes, Dirichlet processes
118 | prob-sota          | prob: compare with scipy.stats, statsmodels, Distributions.jl, R's stats
119 | prob-api           | prob: distribution interface, sampling vs density, vectorized batch
120 | prob-perf          | prob: ziggurat sampling, alias method, batched RNG
121 | queue-numerics     | queue: Erlang-B numerical underflow, Pollaczek-Khinchine accuracy
122 | queue-missing      | queue: missing — M/M/c/K, M/G/c, G/G/1 (Kingman), Jackson networks, BCMP, polling models, fluid limits, fork-join
123 | queue-sota         | queue: compare with simpy, queueing-tool, JMT
124 | queue-api          | queue: closed vs open networks, transient vs steady-state
125 | queue-perf         | queue: Markov-chain matrix exponential, sparse generators
126 | sequence-numerics  | sequence: edit-distance int overflow, fuzzy-match score normalization
127 | sequence-missing   | sequence: missing — Hirschberg LCS, suffix automaton, FM-index, Burrows-Wheeler, Z-algorithm, Aho-Corasick, edit-distance with transpositions, BK-trees
128 | sequence-sota      | sequence: compare with rapidfuzz, jellyfish, edlib, parasail, abPOA
129 | sequence-api       | sequence: byte vs codepoint vs grapheme contract clarity
130 | sequence-perf      | sequence: SIMD bit-parallel Levenshtein (Myers), Wagner-Fischer cache layout
131 | signal-numerics    | signal: FFT roundoff growth, IIR filter quantization, Hilbert envelope causality
132 | signal-missing     | signal: missing — Goertzel, Bluestein, multitaper, wavelets (CWT/DWT), EMD, VMD, Kalman/UKF, Savitzky-Golay, Welch, MUSIC, ESPRIT
133 | signal-sota        | signal: compare with scipy.signal, librosa, GNU Radio, EigenSPL
134 | signal-api         | signal: streaming filter state, sample-rate handling, complex spectra
135 | signal-perf        | signal: Stockham FFT, split-radix, real-FFT optimizations
136 | timeseries-numerics | timeseries: ARIMA likelihood stability, state-space filter conditioning
137 | timeseries-missing | timeseries: missing — SARIMA, ETS, BATS/TBATS, Theta, Prophet, Bayesian structural TS, GARCH(p,q), EGARCH, dynamic linear models, dlm
138 | timeseries-sota    | timeseries: compare with statsmodels, sktime, prophet, Nixtla statsforecast, neuralforecast
139 | timeseries-api     | timeseries: irregular vs regular sampling, multivariate, exogenous regressors
140 | timeseries-perf    | timeseries: state-space filter banded ops, parallel forecast
141 | topology-numerics  | topology: persistence threshold sensitivity, simplicial complex memory
142 | topology-missing   | topology: missing — persistent cohomology, persistence images/landscapes, mapper, Reeb graph, Morse-Smale, persistent local homology, multiparameter persistence, zigzag persistence
143 | topology-sota      | topology: compare with GUDHI, Ripser, Dionysus, giotto-tda, GHB
144 | topology-api       | topology: filtration types, distance functions, output diagrams
145 | topology-perf      | topology: matrix-reduction column algorithms, clearing, twist
146 | zkmark-numerics    | zkmark: field-arithmetic carry handling, Fiat-Shamir transcript discipline
147 | zkmark-missing     | zkmark: missing — KZG, FRI, IPA, PlonK, Halo2, STARK polynomials, Reed-Solomon, multilinear MLE, lookup arguments (logUp, cq, plookup)
148 | zkmark-sota        | zkmark: compare with Arkworks, halo2, gnark, plonky3, sp1, risc0
149 | zkmark-api         | zkmark: circuit DSL ergonomics, witness vs public input
150 | zkmark-perf        | zkmark: NTT/INTT speed, Pippenger MSM, GPU-friendly layouts

## Block B — Cross-package synergies (50)

151 | synergy-signal-prob       | signal × prob: spectral estimation as Bayesian inference, periodogram statistics
152 | synergy-linalg-autodiff   | linalg × autodiff: matrix calculus, Jacobian-vector products, second-order opt
153 | synergy-prob-infogeo      | prob × infogeo: Fisher information from distributions, natural gradient on prob
154 | synergy-chaos-timeseries  | chaos × timeseries: Takens embedding, Lyapunov forecasting, RQA
155 | synergy-crypto-prob       | crypto × prob: randomness extractors, leftover hash lemma, entropy bookkeeping
156 | synergy-topology-prob     | topology × prob: persistence statistics, bottleneck distance bootstrap
157 | synergy-graph-linalg      | graph × linalg: spectral graph theory, normalized Laplacian, heat kernels
158 | synergy-color-signal      | color × signal: image filtering in perceptual spaces, demosaicing
159 | synergy-em-signal         | em × signal: Maxwell→wave equation, dispersion, group/phase velocity
160 | synergy-fluids-signal     | fluids × signal: turbulence spectra, Kolmogorov scaling, POD/DMD
161 | synergy-control-prob      | control × prob: stochastic control, LQG, particle filter
162 | synergy-graph-prob        | graph × prob: random graphs, ERGM, SBM, latent space models
163 | synergy-optim-autodiff    | optim × autodiff: forward-mode for line search, hessian-vector products
164 | synergy-orbital-optim     | orbital × optim: Lambert problem, low-thrust trajectory optimization
165 | synergy-sequence-prob     | sequence × prob: HMMs, profile HMMs, sequence alignment statistics
166 | synergy-acoustics-signal  | acoustics × signal: room acoustics impulse, beamforming, MVDR
167 | synergy-audio-signal      | audio × signal: pitch detection, onset, source separation pipelines
168 | synergy-physics-autodiff  | physics × autodiff: Lagrangian/Hamiltonian neural nets, energy-conserving sim
169 | synergy-prob-optim        | prob × optim: variational inference, EM, MAP via constrained opt
170 | synergy-info-prob         | info × prob: rate-distortion, channel capacity, MI lower bounds
171 | synergy-graph-topology    | graph × topology: clique complex, flag complex, network homology
172 | synergy-changepoint-timeseries | changepoint × timeseries: detection within ARIMA, online segmentation
173 | synergy-queue-prob        | queue × prob: heavy-tailed service, fluid limits, Brownian approximation
174 | synergy-gametheory-optim  | gametheory × optim: best-response dynamics, no-regret learning, online convex
175 | synergy-zkmark-crypto     | zkmark × crypto: pairing-based SNARKs, hash-to-curve, polynomial commitments
176 | synergy-color-prob        | color × prob: gamut sampling, color appearance under uncertainty
177 | synergy-geometry-optim    | geometry × optim: shape optimization, level-set methods, conformal
178 | synergy-control-optim     | control × optim: MPC as online optimization, KKT, active-set
179 | synergy-em-fluids         | em × fluids: MHD, plasma, Hall-MHD primitives
180 | synergy-physics-prob      | physics × prob: Boltzmann distributions, Monte Carlo for stat-mech
181 | synergy-combinatorics-prob | combinatorics × prob: occupancy, urn models, exchangeable sequences
182 | synergy-compression-info  | compression × info: arithmetic coding ↔ entropy, KL bound on compressibility
183 | synergy-calculus-autodiff | calculus × autodiff: when to use AD vs adaptive quadrature, error bounds
184 | synergy-linalg-prob       | linalg × prob: covariance estimation, shrinkage, Ledoit-Wolf, factor models
185 | synergy-signal-autodiff   | signal × autodiff: differentiable DSP, learnable filters
186 | synergy-graph-control     | graph × control: consensus dynamics, multi-agent control, Laplacian flow
187 | synergy-orbital-control   | orbital × control: station-keeping, attitude control quaternion+PID
188 | synergy-prob-linalg       | prob × linalg: low-rank covariance, randomized PCA on samples
189 | synergy-info-compression  | info × compression: minimum description length, BIC/AIC pinning
190 | synergy-topology-signal   | topology × signal: persistent topology of time-series, sliding-window embedding
191 | synergy-chaos-control     | chaos × control: chaos control (OGY method), targeting
192 | synergy-fluids-control    | fluids × control: PI(D) for flow control, drag reduction, ROM-based
193 | synergy-prob-changepoint  | prob × changepoint: Bayesian online changepoint as Markov chain
194 | synergy-em-geometry       | em × geometry: Stokes/Gauss theorems on meshes, exterior calculus
195 | synergy-optim-prob        | optim × prob: SGD as Langevin, stochastic approximation
196 | synergy-color-info        | color × info: ICC profile entropy, perceptual just-noticeable-difference
197 | synergy-acoustics-fluids  | acoustics × fluids: Lighthill analogy, aeroacoustics
198 | synergy-physics-optim     | physics × optim: variational principles as optimization
199 | synergy-graph-info        | graph × info: graph entropy, Shannon capacity, network information
200 | synergy-zkmark-info       | zkmark × info: knowledge soundness, simulator entropy

## Block C — Cutting-edge math not yet in reality (100)

201 | new-optimal-transport     | Optimal transport: Wasserstein, Sinkhorn, entropic regularization, unbalanced OT
202 | new-sde                   | Stochastic differential equations: Itô calculus, Milstein, SRK, MLMC
203 | new-tensor-networks       | Tensor networks: MPS/TT/PEPS, decomposition, contraction order
204 | new-symplectic-int        | Symplectic integrators: Verlet, leapfrog, Yoshida, splitting methods
205 | new-lie-groups            | Lie groups & algebras: SO(3), SE(3), SU(2) operations, exp/log on manifolds
206 | new-riemannian-opt        | Riemannian optimization: trust-region on manifolds, retractions, vector transport
207 | new-diff-geo              | Differential geometry primitives: connections, curvature, parallel transport
208 | new-exterior-calculus     | Exterior calculus: differential forms, Hodge star, de Rham
209 | new-geometric-algebra     | Geometric algebra: rotors, multivectors, conformal GA
210 | new-coding-theory         | Coding theory: Reed-Solomon, BCH, LDPC, polar, turbo
211 | new-lattice-crypto        | Lattice cryptography: LWE, RLWE, NTRU, gadget decomposition
212 | new-pq-signatures         | Post-quantum signatures: Falcon, Dilithium, SPHINCS+, XMSS, LMS
213 | new-isogeny               | Isogeny crypto: SIDH/SIKE remnants, CSIDH, SQISign
214 | new-pairings              | Pairing-based crypto: BLS12-381, BN, Tate/Weil pairings
215 | new-compressed-sensing    | Compressed sensing: RIP, basis pursuit, IHT, AMP
216 | new-rmt                   | Random matrix theory: Wigner, Marchenko-Pastur, Tracy-Widom
217 | new-free-prob             | Free probability: free convolution, R-transform, freeness
218 | new-rough-paths           | Rough path theory: signature, log-signature, controlled paths
219 | new-mean-field-games      | Mean-field games: HJB-Fokker-Planck system, ergodic MFG
220 | new-stochastic-opt        | Stochastic optimization: SAA, SGD with momentum families, variance reduction (SVRG, SAGA)
221 | new-online-learning       | Online learning: regret bounds, mirror descent, Hedge, Online Newton
222 | new-bandits               | Multi-armed bandits: UCB, Thompson, EXP3, contextual, linear, Gaussian process
223 | new-submodular            | Submodular optimization: greedy, continuous, lattice, k-system
224 | new-streaming             | Streaming algorithms: Count-Min, HyperLogLog, t-digest, KLL, MisraGries
225 | new-ann                   | Approximate nearest neighbor: HNSW, IVF-PQ, ScaNN, NSG
226 | new-hyperbolic-embed      | Hyperbolic embeddings: Poincaré ball, Lorentz model, distortion bounds
227 | new-uq                    | Uncertainty quantification: PCE, gPC, stochastic collocation, Sobol indices
228 | new-bayes-nonparam        | Bayesian nonparametrics: Dirichlet process, Indian buffet, Pitman-Yor, HDP
229 | new-causal                | Causal inference: do-calculus, IV, propensity, double ML, DAGs
230 | new-fdr                   | False discovery rate: BH, BY, knockoffs, e-values
231 | new-conformal             | Conformal prediction: split, jackknife+, CV+, weighted, online
232 | new-robust-stats          | Robust statistics: M-estimators, MCD, RANSAC, Huber, S/MM
233 | new-extreme-value         | Extreme value theory: GEV, POT, Hill estimator, return-level
234 | new-copulas               | Copulas: full Archimedean family, vines, R-vines, t, Gaussian
235 | new-functional-data       | Functional data analysis: FPCA, RKHS, smoothing splines
236 | new-rkhs                  | RKHS / kernel methods: kernel mean embedding, MMD, HSIC, kernel ridge
237 | new-gaussian-process      | Gaussian processes: kernels, sparse GP (FITC, VFE), deep GP, multi-output
238 | new-mcmc                  | MCMC: HMC, NUTS, MALA, slice sampling, parallel tempering, NRJ
239 | new-svi                   | Stochastic variational inference: amortized, normalizing flows, IAF, RealNVP
240 | new-normalizing-flows     | Normalizing flows: planar, RealNVP, Glow, neural spline, continuous
241 | new-diffusion-models      | Diffusion / score-based generative: score matching, SDE/ODE form
242 | new-spde                  | Stochastic PDEs: KPZ, SPDE finite elements, mild solutions
243 | new-fpe                   | Fokker-Planck: forward/backward Kolmogorov, finite-volume schemes
244 | new-pde-solvers           | PDE finite-difference / FEM scaffolding: 1D-3D, boundary conditions
245 | new-spectral-methods      | Spectral methods for PDEs: Chebyshev, Fourier, RBF, pseudo-spectral
246 | new-discrete-exterior     | Discrete exterior calculus on simplicial / cubical complexes
247 | new-mortar-fem            | Mortar / non-conforming FEM, hp-adaptivity
248 | new-multigrid             | Multigrid solvers: V/W cycles, FMG, AMG (algebraic)
249 | new-domain-decomp         | Domain decomposition: Schwarz, FETI, BDDC
250 | new-mortar                | Mortar methods, contact, Lagrange multipliers
251 | new-shape-opt             | Shape and topology optimization: SIMP, level-set, phase-field
252 | new-image-segmentation    | Image segmentation primitives: Mumford-Shah, Chan-Vese, graph cut
253 | new-active-contours       | Active contours / snakes, geodesic, Chan-Vese
254 | new-graph-cuts            | Graph cuts: max-flow on grids, α-expansion, QPBO
255 | new-mrf                   | Markov random fields: pairwise, higher-order, message passing
256 | new-belief-propagation    | Belief propagation: sum-product, max-product, generalized BP
257 | new-tensor-decomp         | Tensor decompositions: CP, Tucker, HOSVD, hierarchical Tucker
258 | new-sparse-coding         | Sparse coding / dictionary learning: K-SVD, ISTA, FISTA
259 | new-matrix-completion     | Matrix completion: nuclear-norm, IRLS, iterative SVD
260 | new-robust-pca            | Robust PCA: principal component pursuit, IRLS, R-PCA online
261 | new-online-svd            | Online SVD / streaming PCA, frequent directions
262 | new-randomized-numerics   | Randomized numerical linear algebra: sketches, count-sketch, leverage scores
263 | new-quasi-mc              | Quasi-Monte Carlo: Sobol, Halton, lattice rules, scrambled
264 | new-mlmc                  | Multilevel Monte Carlo: Giles construction, antithetic, randomized
265 | new-pmcmc                 | Particle MCMC: PIMH, particle Gibbs, SMC²
266 | new-smc                   | Sequential Monte Carlo: bootstrap, auxiliary, adaptive resampling
267 | new-abc                   | Approximate Bayesian computation: ABC-rejection, ABC-SMC, MCMC-ABC
268 | new-hmm-extensions        | Hidden Markov extensions: HSMM, IO-HMM, factorial HMM
269 | new-gp-state-space        | GP state-space models: Kalman/EKF/UKF for GPs (Hartikainen-Särkkä)
270 | new-graph-signal-proc     | Graph signal processing: graph Fourier, wavelets, Chebyshev
271 | new-spectral-clustering   | Spectral clustering: normalized cuts, eigengap heuristic
272 | new-manifold-learning     | Manifold learning: ISOMAP, LLE, t-SNE, UMAP, diffusion maps
273 | new-spectral-embedding    | Spectral embedding: Laplacian eigenmaps, locally linear
274 | new-network-flow          | Network flow: min-cost flow, b-matching, transportation
275 | new-matroid               | Matroid intersection / partition / oracle
276 | new-cvxopt-extras         | Convex optimization extras: SDP, SOCP, conic ADMM
277 | new-copo                  | Combinatorial optimization: ILP, branch-and-bound, cutting planes (Gomory)
278 | new-ip-relaxation         | LP/IP relaxations: Lagrangian, branch-and-cut
279 | new-met-h-eta             | Metaheuristics: tabu search, ALNS, GRASP
280 | new-stoch-block-model     | Network models: SBM, degree-corrected SBM, dynamic SBM, latent space
281 | new-temporal-graphs       | Temporal/dynamic graphs: temporal motifs, time-respecting paths
282 | new-hypergraphs           | Hypergraphs: spectral, motifs, Laplacians
283 | new-simplicial-complexes  | Simplicial complexes: as data structure, persistent
284 | new-cw-complexes          | CW-complexes / cell complexes for computation
285 | new-discrete-morse        | Discrete Morse theory, Forman's vector fields
286 | new-reeb                  | Reeb graphs / contour trees / merge trees
287 | new-mapper                | Mapper algorithm for TDA
288 | new-persistence-stats     | Persistence statistics: bottleneck, Wasserstein on diagrams, kernels
289 | new-zigzag                | Zigzag persistence, multi-parameter persistence
290 | new-galois-theory         | Galois theory primitives: field extensions, splitting fields, factoring
291 | new-modular-arithmetic    | Advanced modular arithmetic: CRT in big-int, Montgomery, Barrett, Berlekamp
292 | new-elliptic-curves       | Elliptic curves over finite fields: point counting (SEA), endo, complex multiplication
293 | new-ntt                   | Number-theoretic transform variants: cooley-tukey, harvey, butterfly negacyclic
294 | new-modular-forms         | Modular forms: Eisenstein, theta, q-expansions
295 | new-l-functions           | L-functions, zeta computation primitives
296 | new-generating-functions  | Generating functions: ordinary, exponential, Dirichlet, ops on them
297 | new-asymptotic-analysis   | Asymptotic analysis tools: Watson lemma, saddle-point, steepest descent
298 | new-hypergeometric        | Hypergeometric functions: ₁F₁, ₂F₁, generalized, Mellin-Barnes
299 | new-special-functions     | Special functions: elliptic E/F/Π, Jacobi, theta, Mathieu, parabolic cylinder
300 | new-bessel-spherical      | Bessel/spherical Bessel/Hankel families with high precision

## Block D — Specific deep dives (50)

301 | dive-fft-correctness      | FFT: Cooley-Tukey vs split-radix vs Stockham; numerical roundoff comparison
302 | dive-stable-sums          | Stable summation: Kahan, Neumaier, pairwise; what reality should standardize on
303 | dive-relerr-bounds        | Relative-error bounds: existing functions documented vs reality
304 | dive-rng-quality          | RNG quality: TestU01 BigCrush coverage of reality's PRNGs
305 | dive-bigint-mul           | Big-int multiply: Karatsuba, Toom-Cook, Schönhage-Strassen, Furer
306 | dive-prime-tests          | Primality testing: BPSW, AKS, ECPP, Frobenius
307 | dive-discrete-log         | Discrete log: BSGS, Pollard rho, kangaroo, index calculus where applicable
308 | dive-kalman-square-root   | Square-root Kalman / UD factorization for stability
309 | dive-kalman-info-form     | Information-form Kalman, Bayesian recursion variants
310 | dive-particle-filter      | Particle filter resampling schemes (multinomial, systematic, residual)
311 | dive-gmres-restart        | GMRES restart strategies, deflation, augmented Krylov
312 | dive-bilinear-bias        | Bilinear interpolation bias, cubic, Lanczos, Mitchell-Netravali tradeoffs
313 | dive-rotation-rep         | Rotation representations: quaternion vs DCM vs Euler vs Rodrigues
314 | dive-ahrs                 | AHRS: Madgwick, Mahony, ESKF — what reality should expose
315 | dive-iir-design           | IIR design: bilinear, impulse-invariant, MZT, frequency warping
316 | dive-fir-design           | FIR design: Parks-McClellan, Kaiser, ls, frequency sampling
317 | dive-window-functions     | Window functions catalog: Kaiser, DPSS, Slepian, flat-top, Nuttall
318 | dive-resampling           | Resampling: polyphase, sinc, Lagrange, Farrow, Smith arbitrary
319 | dive-pll                  | PLL primitives: Costas, type-II/III loops, gear-shifted
320 | dive-error-correction     | Error-correcting codes: Reed-Solomon, Hamming, BCH coverage
321 | dive-finite-field         | Finite field GF(2^n) operations: poly mul, inverse, sqrt, irreducible test
322 | dive-aes-vs-poly1305      | Symmetric primitives: AES, ChaCha20, Poly1305, GHASH; what's missing
323 | dive-hash-to-curve        | Hash-to-curve (BLS12-381), encode-to-curve algorithms
324 | dive-msm                  | Multi-scalar multiplication: Pippenger, Bos-Coster, GLV/GLS
325 | dive-poseidon             | Algebraic hashes: Poseidon, Rescue, Griffin, Anemoi, Vision
326 | dive-merkle-trees         | Merkle tree variants: binary, sparse, Verkle, KZG-based
327 | dive-ot-extension         | OT extension: IKNP, KOS, Silver, primitives reality could expose
328 | dive-ad-jvp-vjp           | Forward-mode jvp vs reverse vjp tradeoffs in autodiff
329 | dive-checkpointing        | Checkpointed reverse mode (Griewank): memory/compute tradeoff curves
330 | dive-implicit-diff        | Implicit differentiation through fixed-points and optimizers
331 | dive-ad-fixedpoint        | AD through iterative algorithms: deep equilibrium, fixed-point unrolling
332 | dive-mpc-quad             | MPC: quadratic programming primitives, OSQP-style ADMM
333 | dive-trajopt              | Trajectory optimization: DDP, iLQR, direct collocation
334 | dive-mppi                 | MPPI / model-predictive path integral
335 | dive-cmaes-tuning         | CMA-ES variants: sep-CMA, IPOP, BIPOP, lm-CMA
336 | dive-bo-acquisition       | Bayesian optimization acquisition: EI, UCB, KG, PES, MES, qEI
337 | dive-fft-fractional       | Fractional FFT (chirp z-transform); applications & precision
338 | dive-hilbert-transform    | Hilbert transform implementations: FIR, all-pass IIR, FFT-based
339 | dive-acoustic-hrtf        | HRTF processing primitives: minimum-phase, ITD/ILD extraction
340 | dive-room-image-source    | Image-source method for room IR: numerical accuracy at large order
341 | dive-quaternion-slerp     | Slerp/squad/nlerp; precision comparisons, drift
342 | dive-kepler-conic         | Universal Kepler / Stumpff functions for high-eccentricity
343 | dive-lambert-Izzo         | Lambert solvers: Battin, Gauss, Sun, Izzo's universal
344 | dive-orbital-perturbations | Orbital perturbations: J2 only vs full zonal; SGP4 sources
345 | dive-rational-polynomial  | Rational polynomial / Padé approximation utilities
346 | dive-chebyshev-approx     | Chebyshev / Remez polynomial approximation routines
347 | dive-double-double        | Double-double / quad-double arithmetic for selective high precision
348 | dive-interval-arith       | Interval arithmetic: rounding-mode aware operations, Kahan vs IEEE
349 | dive-floating-summation   | Floating summation algorithms benchmark across libraries
350 | dive-fp-precision-modes   | Reduced/extended precision (bf16, fp16, fp8, fp64) — what reality should accept

## Block E — Internet research themes (30)

351 | research-papers-2025      | Latest 2025 papers across math fields most relevant to reality (web)
352 | research-papers-2026      | 2026 papers / preprints relevant to reality (web)
353 | research-libs-rust        | Rust math ecosystem 2025-26: nalgebra, ndarray, candle, polars
354 | research-libs-julia       | Julia ecosystem comparisons 2025-26: SciML, MTK, JuliaStats
355 | research-libs-python      | Python ecosystem 2025-26: JAX, PyTorch internals, scientific stack
356 | research-libs-cpp         | C++ ecosystem: Eigen, Blaze, xtensor, Kokkos, mdspan
357 | research-libs-go          | Go math ecosystem: gonum, gorgonia, what's missing in Go specifically
358 | research-benchmarks       | Standardized math benchmarks (ScalFMM, BLAS-3, FFTW, suiteSparse)
359 | research-correctness      | Correctness-test corpora: TOMS, SCIPy test suite, R reference, MPFR-precision
360 | research-validation       | Cross-language float validation (e.g. crlibm, RLIBM)
361 | research-icml-neurips     | ICML/NeurIPS 2025 math toolbox advances
362 | research-stoc-focs        | STOC/FOCS recent: improved algorithms relevant to reality
363 | research-soda             | SODA recent: discrete algorithms relevant
364 | research-arxiv-math-na    | arXiv math.NA recent items, last 12 months
365 | research-arxiv-stat       | arXiv stat.ME / stat.ML recent
366 | research-arxiv-quant-ph   | arXiv quant-ph: quantum primitives for classical libs
367 | research-codata           | CODATA 2022 final, ITRF/IAU updates, units (BIPM)
368 | research-iau-frames       | IAU frames updates, ICRF3, planetary ephemerides DE441/440/430
369 | research-codecs           | Modern codecs: AV2, JPEG-XL, FFV1; math primitives behind them
370 | research-zk-2026          | State of ZK in 2026: which proof systems, which curves, what's stable
371 | research-pq-2026          | Post-quantum standards 2026: NIST PQC final, ML-KEM, ML-DSA, FN-DSA
372 | research-cryptopals       | Modern cryptographic competitions and challenges, what primitives win
373 | research-fft-libs         | FFT lib comparison 2026: FFTW, MKL, KFR, oneAPI, pocketfft, mufft
374 | research-blas-modern      | BLAS-modern: BLIS, OpenBLAS, MKL, AOCL — kernels worth porting
375 | research-sparse-modern    | Modern sparse: SuiteSparse:GraphBLAS, ParaSails, kPlex
376 | research-prng-modern      | Modern PRNGs: PCG, xoshiro, romu, splitmix, philox; quality criteria
377 | research-num-stab         | Numerical stability papers (Higham, recent advances)
378 | research-survey-physical  | Physical constants & metrology updates, redefinition impacts
379 | research-rust-fast-math   | Rust 'fast math' research, contracts, missing in Go
380 | research-go-math-extras   | Go-specific math extras: x/exp/constraints, math/big improvements

## Block F — Meta / cross-cutting (20)

381 | meta-types-system         | Type system for math: how reality could express dimensionality, units, manifolds
382 | meta-test-coverage        | Cross-cutting: Test coverage gaps audit (golden + property + fuzz)
383 | meta-property-tests       | Property-based tests inventory; what invariants are unencoded
384 | meta-fuzzing              | Fuzzing surface: which inputs are hardest, what dropped at -fuzz
385 | meta-error-handling       | Error handling philosophy: when to panic, when to return, when to NaN
386 | meta-allocation-discipline | Allocation discipline: which packages are hot-path-clean, which leak
387 | meta-simd-strategy        | SIMD strategy across packages: where it matters, dispatch policy
388 | meta-thread-safety        | Thread safety contracts per package; doc gaps
389 | meta-doc-cohesion         | Documentation cohesion: math notation, citations, when to LaTeX
390 | meta-naming-conventions   | Naming conventions audit: are similar concepts named the same?
391 | meta-units-of-measure     | Units of measure across packages — is constants used uniformly?
392 | meta-time-conventions     | Time conventions (TT, TAI, UTC, TDB) across orbital, physics, signal
393 | meta-frame-conventions    | Frame conventions across orbital, geometry, em
394 | meta-cross-language       | Cross-language test parity: what fails Python/C++/C# validation
395 | meta-vendor-deps          | Vendor & deps audit: any cracks in the zero-deps invariant?
396 | meta-build-targets        | Build-target audit: WASM, mobile, embedded constraints
397 | meta-precision-modes      | Precision modes: should reality accept fp32 or stay fp64?
398 | meta-streaming-vs-batch   | Streaming vs batch APIs: who has both, who doesn't
399 | meta-domain-coverage      | Domain coverage map: which areas of math are most under-served vs goals
400 | meta-synthesis-grand      | Grand synthesis: top-30 highest-leverage additions/improvements across all 399 reviews

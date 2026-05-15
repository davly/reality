# 358 — research-benchmarks (standardized math benchmarks reality should run)

## Headline
At least 14 standardized external benchmark suites map cleanly to reality's 22 packages; building a `bench/` directory that imports their golden vectors would cover ~70% of reality's primitives with third-party validation.

## Benchmark catalog

### 1. HPL / LINPACK (Top500)
Solves a dense `Ax=b` system in double precision via LU with partial pivoting; reports sustained Gflops. Reference impl is Netlib `hpl-2.3` (BSD-style); November 2025 Top500 leader is El Capitan at 1.809 Eflop/s. Reality slot: `linalg` LU/Cholesky paths, `linalg.MatMul`. reality is not distributed and won't compete on raw flops, but a single-node HPL run with N≈10k would pin LU correctness across a wide problem and demonstrate cache-blocking vs naive Gflops. Pass criterion: residual `||Ax-b||/(||A||·||x||·N·eps) ≤ 16` per HPL spec. Free download, no license barrier.

### 2. HPCG (High-Performance Conjugate Gradient)
Sparse `Ax=b` with 27-pt stencil + symmetric Gauss-Seidel preconditioner; stresses memory bandwidth, not flops. November 2025 leader: El Capitan at 17.41 HPCG-Pflop/s (vs 1809 HPL Pflop/s — ~1% efficiency, illustrating real-app gap). Reference: `hpcg-benchmark/hpcg` GitHub (BSD). Reality slot: `linalg.SparseCG`, `linalg.SymGS`. The 27-pt stencil generator is trivially portable to Go; the published validation residual norm is the golden value. Excellent fit because reality's sparse CG has no canonical performance baseline today.

### 3. STREAM (McCalpin memory bandwidth)
Four kernels — Copy, Scale, Add, Triad — over arrays ≥4× last-level cache. Measures sustained DRAM bandwidth in MB/s. Reference: `cs.virginia.edu/stream/` (free, no license). Not a correctness test but reality's `linalg.Axpy`-style buffer-output APIs (per CLAUDE.md "no allocations in hot paths") should be benchmarked with STREAM Triad as a ceiling: any reality vector op that does <60% of STREAM Triad on the same array size has room to improve. Direct hook: `linalg.Vec.AddScaled`, `signal.Convolve`.

### 4. FFTW benchFFT
Comprehensive accuracy + speed benchmark of FFT implementations across 1D/2D/3D, real/complex, power-of-2 and prime sizes. Source: `github.com/FFTW/benchfft`. Accuracy results at `fftw.org/accuracy/` (relative L2 error vs MPFR reference). Reality slot: `signal.FFT`, `signal.IFFT`, `signal.RealFFT`. Pass criterion: relative error ≤ ~1e-13 for double-precision power-of-2 sizes ≤ 2^20. Reality's FFT is Cooley-Tukey radix-2; benchFFT would expose any non-power-of-2 weakness (Bluestein/Rader gaps). License: GPL for FFTW itself, but benchFFT inputs/outputs are reproducible from spec.

### 5. SuiteSparse Matrix Collection
1356+ real-world sparse matrices (sparse.tamu.edu) used as standardized inputs for sparse solvers. License: each matrix has its own (mostly research-friendly); collection metadata is open. Reality slot: `linalg` sparse paths, `graph` (matrices induce graphs). For each matrix, golden values are: nnz, condition estimate, CG iteration count to 1e-9 residual on the symmetric set. Pulling 20 well-known matrices (e.g., bcsstk*, nasa*, west*) into `testutil` golden files would pin sparse-solver correctness against published results. Cross-language validation (Go/Py/C++/C#) is straightforward via Matrix Market `.mtx` format.

### 6. ScalFMM
Inria Fast Multipole library (C++, gitlab.inria.fr/solverstack/ScalFMM). Reference benchmarks for N-body Coulomb / 1/r kernel with N up to 10^8. License: LGPLv3. Reality has no FMM today — slot 358's FMM relevance is aspirational: if reality adds an `nbody` package (Barnes-Hut / FMM), ScalFMM's Chebyshev-interpolation accuracy tables would be the validation target. Pass criterion: relative L∞ error ≤ 1e-6 for order-7 interpolation. Practical near-term use: validate `physics.GravityForce` pairwise sums against ScalFMM's brute-force baseline up to N=10^4.

### 7. BBOB / COCO (Black-Box Optimization Benchmarking)
24 noiseless single-objective functions (Sphere, Ellipsoid, Rastrigin, Rosenbrock, Schwefel, …) at 6 dimensions {2,3,5,10,20,40}, 15 instances each = 2160 problem instances. Reference: `numbbo/coco` (BSD-3). Pass criterion: ECDF (empirical cumulative distribution of runtimes) curves comparable across optimizers; median target is `f_opt + 1e-8`. Reality slot: `optim.NelderMead`, `optim.LBFGS`, `optim.SimulatedAnnealing`, `optim.GeneticAlgorithm`, `optim.Simplex`. **This is the highest-leverage benchmark for reality** — `optim` has 6+ algorithms with no comparative published numbers. COCO post-processor produces publication-quality plots.

### 8. BBOB-large-scale
Extension of BBOB to dimensions 20–640 with permuted-block transformations to keep problems non-separable but tractable. arXiv:1903.06396. Reference: `numbbo/coco` same repo (BSD). Reality slot: `optim.LBFGS` shines here (quasi-Newton scales, Nelder-Mead does not). Pass criterion: same ECDF methodology, dimensionally scaled budgets. Pulling the f1–f24 generator code into a Go port (or shelling to Python COCO) gives reality immediate large-D evidence.

### 9. DACBench (Dynamic Algorithm Configuration)
`automl/DACBench` — benchmarks for hyperparameter schedules (CMA-ES step-size, SGD learning rate). License: Apache-2.0. arXiv:2105.08541. Reality slot: tangential — reality is a math library, not an RL/AutoML system. **Recommendation: out of scope** unless reality adds an `optim.AdaptiveSchedule` module. Listed for completeness; do not invest.

### 10. TestU01 (BigCrush PRNG battery)
106 statistical tests, ~5h on a high-end CPU; consumes ~10^11 32-bit samples. Reference: L'Ecuyer & Simard 2007, simul.iro.umontreal.ca/testu01 (free). Pass criterion: no test reports p-value outside [0.001, 0.999]. Reality slot: `crypto.PRNG`, `crypto.LCG`, `crypto.Xoshiro`, `crypto.PCG`, `prob.Sample*`. Lemire's `lemire/testingRNG` repo has BigCrush results for ~30 PRNGs as published baseline. Reality should run BigCrush in CI nightly (not per-commit) and store per-PRNG pass/fail tables in golden files. 32-bit input limit is a known caveat; 64-bit version is in progress upstream.

### 11. NIST SP 800-22 rev1a
15 statistical tests for cryptographic PRNGs (frequency, runs, DFT, linear-complexity, Maurer's universal, …). NIST publishes both spec and C reference impl (free, public domain). Reality slot: `crypto.SecureRandom`, `crypto.HashRNG`. Pass criterion: per-test p-value ≥ 0.01 across ≥ 1000 sequences of 10^6 bits each. Lighter than BigCrush; suitable for per-PR CI. Failures here would invalidate any cryptographic claim — currently reality makes none, but if `crypto.AESCTR_DRBG` lands, this is the gate.

### 12. DLMF Standard Reference Tables (NIST)
"Standard Reference Tables on Demand" — for each special function the DLMF provides high-precision (typically 25–50 digit) test values at chosen arguments. dlmf.nist.gov, public domain (US gov work). Reality slot: anything special — currently `prob` (gamma, beta, erf), `physics` (Bessel if added), `signal` (sinc, window functions). Pass criterion: relative error ≤ 1e-13 vs DLMF tabulated value. **This is the most important slot-290–300 benchmark**: harvest DLMF tables for Γ, ψ, B, erf, erfc, J_n, Y_n, K_n, Ei, Si, Ci, Ai, Bi, ζ, Li_n into golden files. The DLMF Mathematical Introduction explicitly intends these for implementation testing.

### 13. Vallado-Crawford-Hujsak-Kelso 2006 SGP4 vectors (AIAA 2006-6753)
"Revisiting Spacetrack Report #3 Rev 2" — the canonical SGP4 propagator test set. CelesTrak distributes `AIAA-2006-6753.zip` (free) with reference TLEs and propagated state vectors at multiple epochs. Reality slot: `orbital` (currently Kepler-only — does not implement SGP4). If reality adds `orbital.SGP4`, this dataset is the only acceptable validator: ~33 test cases including deep-space resonant orbits, decaying objects, and lunar/solar perturbation cases. Pass criterion: position error < 1 km vs reference at TLE epoch + 1 day. Fortran reference and modern C/Python ports are public.

### 14. Lambert problem benchmarks (Curtis 5.2, Vallado 7-5, lamberthub)
Curtis "Orbital Mechanics for Engineering Students" §5.2 and Vallado "Fundamentals of Astrodynamics" Example 7-5 are the textbook gold standards (single Earth-orbit case, Δv to ~1e-3 km/s precision). Lamberthub (`jorgepiloto/lamberthub`, GPL-3) provides 28 test cases covering single-rev, multi-rev, parabolic, and hyperbolic transfers across multiple solver families (Gooding, Izzo, Battin, Lancaster-Blanchard). Reality slot: `orbital.Lambert` (if added). Pass criterion: terminal-velocity match within solver tolerance (1e-8 km/s).

### 15. LMFDB L-functions / modular forms
Database of L-function zeros, Dirichlet coefficients, and special-value verifications at lmfdb.org (CC-BY-SA). Reality slot: none currently — `crypto` has number theory but no L-function machinery. Listed because slot 295 references it: if reality adds `numtheory.RiemannZeta` beyond the strip currently covered, LMFDB's first-N nontrivial zeros (computed to >1000 digits by Odlyzko / Platt) are the validation target. Pass criterion: zero ordinates match to 10 digits for first 10^4 zeros. Out of scope short-term; document as future hook.

### 16. ZK-proof benchmarks (zk-Bench, Halo2/Plonk/Groth16)
zk-Bench (eprint.iacr.org/2023/1503) compares proof size, prover time, verifier time across Groth16 (192-byte proof, ~3ms verify), PLONK (universal setup, 2.25–9.4× slower prove), Halo2 (no trusted setup, recursive). Reality slot: out of scope — reality has primality, modular arithmetic, hashes, but no full SNARK. **Listed only for completeness**; building a SNARK belongs in a different repo than reality. The number-theory primitives (`crypto.ModExp`, `crypto.MillerRabin`) could be benchmarked against gnark/arkworks as a sanity check but that's not what zk-Bench measures.

### 17. Berkeley TestFloat / IBM FPgen / IeeeCC754++
TestFloat (Hauser, jhauser.us) compares an FPU against SoftFloat reference. IBM FPgen (research.ibm.com/haifa) and IeeeCC754++ provide thousands of edge-case vectors covering subnormals, ±0, ±Inf, NaN payloads, and all 5 IEEE rounding modes. License: free for research. Reality slot: every package — CLAUDE.md mandates "IEEE 754 edge cases mandatory: +Inf, -Inf, NaN, -0.0, subnormals". Today reality embeds these per-function in golden files; importing the IBM FPgen suite (~80k vectors for add/sub/mul/div/sqrt/fma/conv) would dwarf the current per-function 30-vector minimum and give true conformance evidence. Note Go's `math` package itself is the constraint, not reality — but reality is what users will trust.

### 18. HEP-style benchmarks (ROOT-bench, GeantV)
CERN's ROOT framework uses physics-validation suites for kinematics, lorentz transforms, MC integration. License: LGPL. Reality slot: `physics` Lorentz transform if added; `prob.MonteCarlo`. Marginal fit — reality is not particle-physics-flavored. Skip unless `physics.Relativity` lands. Listed because the user prompt named it; recommend deferring.

## Reality benchmark coverage matrix

| Reality area | External benchmark | Coverage today | Coverage if `bench/` added |
|---|---|---|---|
| Dense linalg (`linalg`) | HPL, STREAM | none | full LU + bandwidth ceiling |
| Sparse linalg (`linalg`) | HPCG, SuiteSparse | none | 27-pt + 20 SuiteSparse matrices |
| FFT (`signal`) | FFTW benchFFT | partial (golden vectors) | accuracy + speed plots |
| Optimization (`optim`) | BBOB, BBOB-LS, COCO | none | 24×6×15 instances, ECDF curves |
| PRNG / hash (`crypto`) | TestU01 BigCrush, NIST SP 800-22 | none | full 106-test BigCrush + 15-test NIST |
| Special functions (`prob`, `physics`) | DLMF tables | partial (gamma, erf golden) | full Γ, ψ, B, J_n, Y_n, Ei, ζ, Ai, Bi |
| IEEE 754 conformance | TestFloat, IBM FPgen | per-fn 30 vectors | 80k vector import |
| Astrodynamics (`orbital`) | SGP4 vectors, Lambert (Curtis/Vallado) | Kepler only | depends on SGP4/Lambert add |
| FMM / N-body | ScalFMM | none | aspirational |
| Number theory / SNARK | LMFDB, zk-Bench | none | out of scope |
| HEP physics | ROOT-bench | none | out of scope |
| AutoML | DACBench | none | out of scope |

**Score:** of 22 reality packages, ~14 have at least one published external benchmark; ~9 have multiple. After importing the in-scope suites, reality would have third-party validation for `linalg`, `signal`, `optim`, `crypto`, `prob`, `physics` (special-fn portion), `orbital` (after SGP4/Lambert), and full IEEE-754 conformance — covering well over half the test surface.

## Recommendation

1. **Create `bench/` top-level directory** with subdirs `bench/bbob/`, `bench/hpcg/`, `bench/fftw/`, `bench/testu01/`, `bench/nist80022/`, `bench/dlmf/`, `bench/suitesparse/`, `bench/sgp4/`, `bench/lambert/`, `bench/ieee754/`. Each subdir contains a Go test that pulls golden vectors from the canonical source URL into `testdata/` (cached, hash-pinned) and runs reality's primitive against them.
2. **Highest ROI first:** DLMF (special functions, drop-in golden values) → BBOB (`optim` has nothing today) → IBM FPgen (multiplies test count 50×) → NIST SP 800-22 (cheap, gates any future `crypto` claim).
3. **Defer:** ScalFMM (no FMM in reality), ZK benchmarks, LMFDB, DACBench, HEP — all need new packages first.
4. **CI tiers:** per-PR runs the cheap suites (NIST SP 800-22 short, DLMF, IEEE-754); nightly runs BigCrush, BBOB full, HPCG, SuiteSparse top-50.
5. **License hygiene:** all in-scope suites are MIT/BSD/public-domain/CC-compatible. Lamberthub is GPL-3; cite results from it but reimplement test cases from Curtis/Vallado textbooks (PD facts) to keep reality MIT-clean.
6. **Cross-language validation:** reality's existing 4-language testutil pattern fits naturally — BBOB and DLMF vectors can be regenerated in Go via `math/big` at 256-bit precision and cross-checked against Python `coco` and `mpmath`/`scipy.special` references, which is exactly the existing golden-file workflow.

## Sources
- [COCO: BBOB test suite](https://coco-platform.org/testsuites/bbob/overview.html)
- [numbbo/coco GitHub](https://github.com/numbbo/coco)
- [BBOB-large-scale arXiv:1903.06396](https://arxiv.org/abs/1903.06396)
- [TestU01 (L'Ecuyer & Simard, Université de Montréal)](https://simul.iro.umontreal.ca/testu01/tu01.html)
- [lemire/testingRNG BigCrush results](https://github.com/lemire/testingRNG)
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- [FFTW benchFFT](https://www.fftw.org/benchfft/)
- [FFTW accuracy results](https://www.fftw.org/accuracy/)
- [HPCG benchmark site](https://www.hpcg-benchmark.org/)
- [HPCG TOP500 Nov 2025](https://top500.org/lists/hpcg/2025/11/)
- [HPL on Netlib](https://www.netlib.org/benchmark/hpl/)
- [TOP500 Nov 2025 highlights](https://www.top500.org/lists/top500/2025/11/highs/)
- [STREAM benchmark (McCalpin)](https://www.cs.virginia.edu/stream/)
- [DLMF — NIST Digital Library of Mathematical Functions](https://dlmf.nist.gov/)
- [Vallado et al. AIAA 2006-6753 Revisiting Spacetrack Report #3](https://celestrak.org/publications/AIAA/2006-6753/)
- [lamberthub (Lambert solvers)](https://github.com/jorgepiloto/lamberthub)
- [LMFDB L-functions database](https://www.lmfdb.org/)
- [NIST SP 800-22 rev1a](https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf)
- [Berkeley TestFloat (Hauser)](http://www.jhauser.us/arithmetic/TestFloat.html)
- [IBM FPgen IEEE 754 test suite](https://github.com/sergev/ieee754-test-suite)
- [DACBench (automl)](https://github.com/automl/DACBench)
- [zk-Bench eprint.iacr.org/2023/1503](https://eprint.iacr.org/2023/1503.pdf)
- [ScalFMM (Inria)](https://solverstack.gitlabpages.inria.fr/ScalFMM/index.html)

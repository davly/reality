# 263 | new-quasi-mc — Quasi-Monte Carlo: Sobol, Halton, lattice rules, scrambled

**Summary line 1.** reality v0.10.0 ships effectively **zero** public quasi-Monte-Carlo
surface — repo-wide grep on `Sobol|Halton|Faure|Niederreiter|Korobov|VanDerCorput|van.der.Corput|RadicalInverse|LowDiscrepancy|low.discrepancy|StarDiscrepancy|L2Discrepancy|KoksmaHlawka|Koksma.Hlawka|LatticeRule|Polynomial.Lattice|Owen.Scramble|OwenScramble|Digital.Net|t.s.net|t,m,s.net|Latin.Hypercube|LatinHypercube|LHS|Stratified.Sample|StratifiedSample|Effective.Dimension|EffectiveDimension|RQMC|Randomized.QMC|HigherOrder.QMC|Padding.QMC` returns **one** Go callable: the **8-LOC private helper** `halton(n, base int) float64` at `audio/separation/nmf.go:233` (a textbook van-der-Corput / Halton radical-inverse used as a deterministic non-zero NMF initialiser to dodge the multiplicative-update zero-fixed-point); zero callable matches anywhere else in the 22-package surface — `calculus/calculus.go:244 MonteCarloIntegrate(f, dim, lower, upper, samples, rng)` is **vanilla** uniform-random MC with `O(N^{-1/2})` convergence, no QMC mode, no scrambling, no LHS, no stratification; `crypto/rng.go` ships `MersenneTwister` + `PCG` + `Xoshiro256` (per slot 175) but they emit **uniform pseudorandom** not low-discrepancy — wrong tool entirely; `prob/distributions.go` has CDFs/quantiles for inverse-CDF transformation but **no public sampler** (gates Gaussian-QMC since CDF⁻¹ ∘ Halton is the standard Gaussian-QMC realisation, also missing); `combinatorics/` has zero QMC surface (037 enumerated 14 missing primitives, none Halton/Sobol). The 1960-2024 QMC canon is among the most consequential unfulfilled promises in the repo: Halton-1960-Numer-Math-2:84 *On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals* (~2,800 citations — the foundational 1960 paper; radical-inverse φ_b(n) = sum d_i b^{-i-1} where n = sum d_i b^i in base b, d-dim Halton uses first-d-primes basis); Faure-1982-Acta-Arith-41:337 *Discrépance de suites associées à un système de numération* (~1,200 citations — Faure-sequence: prime base ≥ d, all dims share base via permutation matrix, lower-discrepancy than Halton for d ≤ ~30); Sobol-1967-USSR-Comput-Math-1:86 *On the distribution of points in a cube and the approximate evaluation of integrals* (~7,500 citations — direction numbers v_{j,k} with primitive polynomials, base-2 Gray-code recursion `x_{n+1} = x_n XOR v_{c(n)}` where c(n) is the position of the rightmost zero in n; the **single most-cited QMC sequence** by ~3x; Joe-Kuo-2008-SIAM-J-Sci-Comput-30:2635 *Constructing Sobol sequences with better two-dimensional projections* ships canonical direction-number tables up to dim 21,201, the production-default since 2008); Niederreiter-1987-J-Number-Theory-30:51 + Niederreiter-1992-CBMS-63 *Random Number Generation and Quasi-Monte Carlo Methods* (~3,200 citations — `(t, m, s)`-net + `(t, s)`-sequence general framework subsuming Halton-Sobol-Faure; t = quality parameter, lower is better, t=0 is optimal but rare); Owen-1995-Stat-Sinica + Owen-1997-Ann-Stat-25:1541 + Owen-2003-SIAM-J-Numer-Anal-41:1837 *Variance with alternative scramblings of digital nets* (~1,800 citations across the trilogy — **nested-uniform-scrambling / Owen-scrambling**: digit-by-digit random permutation that randomises a digital net without breaking its `(t,m,s)`-net structure, gives unbiased estimator + O(N^{-3/2} (log N)^{(s-1)/2}) RMSE for smooth integrands vs Sobol's O(N^{-1} (log N)^s)); McKay-Beckman-Conover-1979-Technometrics-21:239 *A comparison of three methods for selecting values of input variables in the analysis of output from a computer code* (~9,500 citations — **Latin Hypercube Sampling**: stratified-1D-marginals + random-pairing across dims, the single most-used "almost-QMC" method in engineering); Korobov-1959-Doklady-Akad-Nauk-SSSR-124:1207 *Number-theoretic methods in approximate analysis* (~600 citations — **rank-1 lattice rule**: `x_k = (k/N) z mod 1` for generating-vector `z ∈ Z^d`, periodic-integrand QMC with O(N^{-2}) for analytic integrands); Sloan-Joe-1994-Oxford-Lattice-Methods-for-Multiple-Integration (~1,500 citations — modern lattice-rule textbook); Sloan-Wozniakowski-1998-J-Complexity-14:1 *When are quasi-Monte Carlo algorithms efficient for high dimensional integrals?* + Sloan-Kuo-Joe-2002-Math-Comp-71:1609 + Kuo-Sloan-2005-Notices-AMS-52:1320 (component-by-component CBC construction for randomly-shifted lattice rules with weighted Sobolev space tractability); Caflisch-Morokoff-Owen-1997-J-Comput-Finance-1:27 *Valuation of mortgage-backed securities using Brownian bridges to reduce effective dimension* (~1,400 citations — **effective dimension**: nominal-d may be 360 but truncation-d ≤ 5 explains 99% of variance; Brownian-bridge / PCA / Gaussian-Sobol to align principal axes with leading QMC dims, the keystone *finance-QMC* technique); Dick-2008-Ann-Stat-36:2429 + Dick-Pillichshammer-2010-Cambridge-Digital-Nets-and-Sequences (~3,000 citations textbook — **higher-order QMC** via interlaced digital nets achieving O(N^{-α} (log N)^{ds}) for α-smooth integrands beating the O(N^{-1}) classical Sobol limit). The closest tangential surfaces in reality are the **8-LOC private Halton helper** `audio/separation/nmf.go:halton` (must be lifted to a public sibling), the **`MonteCarloIntegrate`** signature in `calculus/calculus.go:244` (the natural injection-point: replace the Float64()-RNG argument with a `Sequence` interface), the **CDF-quantile families** in `prob/distributions.go:32-260` (the inverse-CDF transformation that turns uniform-QMC into Gaussian/Gamma/Beta-QMC), and the **PRNGs** in `crypto/rng.go` (used to seed the Owen-scrambling random permutations, NOT to seed the QMC sequence itself).

**Summary line 2.** Twenty-six primitives **Q1–Q26** totalling **~3,420 LOC pure connective tissue + ~80 LOC NEW substrate** (the same `prob/random/normal.go ~80 LOC` substrate-pool blocking 117/202/215/227/259/260/261/262 — co-shipping amortises) split across **(a) ~480 LOC Tier-1 sequences keystone** in new sub-package `qmc/` (Q1 `qmc/halton.go::Halton(n, dim int, out []float64)` ~80 LOC public lift of the `audio/separation` private — radical-inverse φ_{p_i}(n) into the i-th first-d-primes for i ∈ [1,d]; auto-Halton-degeneracy-warning when dim > 12 due to the visible 2D-projection cluster known as "Halton's path-degeneracy" between high primes; Q2 `qmc/sobol.go::Sobol(n, dim int, out []float64)` ~180 LOC Joe-Kuo-2008 direction-number tables compiled into a static `var jkDirection [maxDim][maxBits]uint32` ~30 KB (Joe-Kuo's canonical "new-joe-kuo-6.21201" file truncated at dim 1024) + Antonov-Saleev-1979-USSR-Comput-Math-19:252 Gray-code recursion `x_{n+1} = x_n XOR v_{c(n)}` — **the single most-cited QMC primitive**, ~7,500 citations; Q3 `qmc/sobol.go::SobolSequence` stateful struct supporting `.Next(out []float64)` and `.Skip(n)` for streaming N=2^k samples without the O(N·d) bulk allocation; Q4 `qmc/vandercorput.go::VanDerCorput(n, base int) float64` ~12 LOC the 1-D building-block of Halton, also useful as a standalone for QMC-univariate-integration where Sobol is overkill — closes the documented private-helper duplicate at `audio/separation/nmf.go:halton`; Q5 `qmc/faure.go::Faure(n, dim int, out []float64)` ~120 LOC Faure-1982 prime-base ≥ d with binomial-permutation transform `x_d = C ⋅ x_1` over GF(p) — better discrepancy than Halton for moderate dim ≤ 30; Q6 `qmc/niederreiter.go::Niederreiter(n, dim int, out []float64)` ~80 LOC Niederreiter-Xing base-2 (t,s)-sequence for dims where Sobol's t-quality degrades — research-grade alternative; Q7 `qmc/discrepancy.go::StarDiscrepancy(points [][]float64) float64` ~80 LOC Heinrich-1996 / Thiémard-2001 deterministic O(N^{2d}) star-discrepancy D*_N — the canonical QMC quality metric used in Koksma-Hlawka), **(b) ~440 LOC Tier-2 lattice rules** (Q8 `qmc/lattice.go::KorobovLattice(n int, generator []int, out [][]float64)` ~80 LOC Korobov-1959 rank-1-lattice `x_k = {k/N · z}` for user-supplied generating-vector z ∈ Z^d; Q9 `qmc/lattice.go::RandomShiftedLattice(rule, shift []float64, out [][]float64)` ~40 LOC Cranley-Patterson-1976-SIAM-J-Numer-Anal-13:904 random-shift `x_k = {x_k^lattice + Δ}` for unbiased RQMC estimator; Q10 `qmc/cbc.go::CBCConstruction(N, d int, weights []float64) []int` ~200 LOC component-by-component construction Sloan-Kuo-Joe-2002 for product-weighted Sobolev space — finds an optimal generating vector z minimising worst-case-error for given N and weights γ_j without exhaustive search; Q11 `qmc/lattice.go::PolynomialLattice(n int, polyGenerator [][]uint8, out [][]float64)` ~120 LOC Niederreiter-1992 + Dick-Kuo-Sloan-2013 polynomial-lattice in F_2[x] — base-2 alternative to Korobov with stronger weighted-tractability guarantees), **(c) ~480 LOC Tier-3 scrambling + RQMC** (Q12 `qmc/scramble.go::OwenScramble(seq Sequence, rng *crypto.Xoshiro256) ScrambledSequence` ~180 LOC Owen-1995 / Owen-1997 / Owen-2003 nested-uniform-scrambling — for each digit position k of each dim j, draw a random permutation σ_{j,k}: {0,...,b-1} → {0,...,b-1} where the permutations are **nested** (σ_{j,k+1} depends on σ_{j,k}); preserves the (t,m,s)-net property; **the single most-consequential RQMC primitive** ~1,800 citations; achieves O(N^{-3/2} log^{(s-1)/2} N) RMSE for smooth integrands; Q13 `qmc/scramble.go::DigitalShift(seq Sequence, rng *crypto.Xoshiro256) ScrambledSequence` ~40 LOC Cranley-Patterson digital-shift in base 2 — XOR each digit with a random per-dim mask; cheaper than Owen-scrambling but only achieves O(N^{-1} log N) variance; Q14 `qmc/scramble.go::LinearMatrixScramble(seq Sequence, rng *crypto.Xoshiro256) ScrambledSequence` ~120 LOC Matoušek-1998-J-Complexity-14:527 random-linear-matrix scrambling — multiply digit-vector by random lower-triangular-with-unit-diagonal matrix L_j over F_2, cheaper than Owen with similar variance; Q15 `qmc/rqmc.go::RQMC(integrand, dim int, samples int, replications int, sequence Sequence, rng *crypto.Xoshiro256) (mean, stderr float64)` ~80 LOC randomised-QMC estimator with R independent scramblings to obtain unbiased mean + std-error confidence interval — closes the gap between QMC's-no-error-bar deterministic-bias and MC's-CLT confidence-interval; Q16 `qmc/scramble.go::OwenScrambleStreaming` ~60 LOC streaming-Owen via on-demand per-digit permutation hashing (Hong-Hickernell-2003) — avoids O(N·d·log_b(N)) permutation table memory for large N), **(d) ~360 LOC Tier-4 sampling primitives consuming QMC** (Q17 `qmc/lhs.go::LatinHypercubeSample(n, dim int, rng *crypto.Xoshiro256, out [][]float64)` ~60 LOC McKay-Beckman-Conover-1979 — for each dim j, partition [0,1] into n equal strata, place one sample per stratum with random per-stratum jitter, then random-permute the inter-dim pairing; the single most-used "almost-QMC" technique in engineering UQ; **CROSS-LINK to 227-U13** ship-once; Q18 `qmc/lhs.go::OptimalLHS(n, dim int, rng *crypto.Xoshiro256, criterion string, out [][]float64)` ~120 LOC Stein-1987 + Park-1994 + Morris-Mitchell-1995-J-Stat-Plan-Inference-43:381 — optimise the LHS by simulated-annealing on a space-filling criterion (maximin / minimax / φ_p / S-criterion); Q19 `qmc/lhs.go::LatinHypercubeOrthogonal(n, dim, q int, rng) [][]float64` ~80 LOC Tang-1993-J-Amer-Stat-Assoc-88:1392 **OA-LHS** orthogonal-array-based LHS combining LHS marginals with strength-2 OA between-dim independence guarantees; Q20 `qmc/stratified.go::StratifiedSample(n, dim int, rng) [][]float64` ~40 LOC simple n=k^d full-grid stratified sampling for low-d (d ≤ 4); Q21 `qmc/inverse.go::InverseCDFSample(uniform [][]float64, distributions []prob.Distribution, out [][]float64)` ~60 LOC standard inverse-CDF transformation U → F^{-1}(U) lifting any uniform-QMC sequence into a non-uniform-QMC sequence — the canonical Gaussian-QMC realisation `Φ^{-1}(Sobol)` for finance; **CROSS-LINK to 227-U13** unblocks Bayesian-QMC and RQMC-finance), **(e) ~360 LOC Tier-5 effective-dimension + variance-reduction** (Q22 `qmc/effective.go::EffectiveDimension(integrand, dim, samples int, criterion string) int` ~120 LOC Caflisch-Morokoff-Owen-1997 — d_T (truncation-dim) the smallest k such that variance from inputs > k is < ε of total; d_S (superposition-dim) the largest interaction-order with non-negligible variance; computed via Sobol-indices ANOVA decomposition (cross-link to 227-U10/U11); Q23 `qmc/brownian.go::BrownianBridge(uniform [][]float64, dt, T float64, out [][]float64)` ~60 LOC Caflisch-Morokoff-Owen-1997 + Moskowitz-Caflisch-1996 path-construction reordering: build Brownian-motion path B_0, B_T/2, B_T/4, B_3T/4, ... using the leading QMC dimensions for the "important" mid-path values not the chronological ones — collapses 360-dim Asian-option to ~5 effective dim; **the keystone finance-QMC technique**; Q24 `qmc/brownian.go::PCAConstruction(uniform [][]float64, cov [][]float64, out [][]float64)` ~60 LOC Acworth-Broadie-Glasserman-1998 PCA-of-covariance-matrix path-construction — alternative to Brownian-bridge using leading eigenvectors of the path covariance; Q25 `qmc/multilevel.go::MultilevelQMC(integrand, levels []int, samples [][]float64, sequence Sequence) (mean, stderr float64)` ~120 LOC Giles-2008 MLMC + Kuo-Schwab-Sloan-2012-Found-Comp-Math-13:1245 **multilevel QMC** — combine cheap-coarse-many-samples with expensive-fine-few-samples using QMC at each level; CROSS-LINK to 202-S5 MLMC and 227-U24 multi-fidelity), **(f) ~300 LOC Tier-6 hybrids + acceleration** (Q26 `qmc/hybrid.go::ScrambledLHS(n, dim int, rng) [][]float64` ~80 LOC Owen-1992 hybrid combining LHS marginal-stratification with Owen-scrambling between-dim — best-of-both-worlds for moderate-d; Q27 `qmc/qmcmc.go::QMCMC(target prob.Distribution, dim, samples int, sequence Sequence) [][]float64` ~120 LOC Chen-Dick-Owen-2011-Ann-Stat-39:673 **QMC-Markov-Chain Monte-Carlo** — replacing the uniform-MCMC accept-reject draw with a CUD (Completely Uniformly Distributed) sequence achieves O(N^{-1+ε}) ergodic-mean convergence beating MCMC's O(N^{-1/2}); CROSS-LINK to 169-MAP-via-LBFGS and 227-U25 Bayesian-Optimisation; Q28 `qmc/importance.go::QMCImportanceSampling(integrand, biasingDist, dim, samples int, sequence Sequence) (estimate, stderr float64)` ~60 LOC importance-sampling driven by inverse-CDF-of-QMC-sequence — variance-reduction stacked on QMC-bias-reduction, the standard derivative-pricing technique; Q29 `qmc/padding.go::PadSequence(low Sequence, high Sequence, totalDim int, splitDim int) Sequence` ~40 LOC Sobol's-trick for d > maxDim of any single sequence: use Sobol for the first 21,201 dims and Halton-on-large-primes for the rest, paying the Halton degeneracy cost only on truly-marginal dims).

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for QMC surface (verified file-walk):

| Surface | Path | Lines | QMC relevance |
|---|---|---:|---|
| `halton(n, base int) float64` (private) | `audio/separation/nmf.go:233` | 8 | **THE ONLY Halton/VdC implementation in the entire repo.** Pure radical-inverse, base-b. Used as deterministic NMF non-zero initialiser. **Must be lifted to public `qmc/halton.go` Q4.** |
| `MonteCarloIntegrate(f, dim, lower, upper, samples, rng)` | `calculus/calculus.go:244` | ~30 | Vanilla uniform-RNG MC, no QMC mode. **Natural injection-point: replace `rng` with `Sequence` interface to enable QMC integration with one-line caller-side change.** |
| `MersenneTwister`, `PCG`, `Xoshiro256` PRNGs | `crypto/rng.go` | 195 | Substrate-RNG for the Owen-scramble random permutations and for LHS jitter. **NOT a QMC sequence** — emits uniform pseudorandom not low-discrepancy. |
| `NormalCDF`, `NormalQuantile` (Acklam) | `prob/distributions.go:47, :67` | ~50 | **DIRECT BUILDING-BLOCK** for Gaussian-QMC via inverse-CDF transformation Φ⁻¹(QMC). Already accurate to ~1.15e-9. |
| `BetaCDF`, `GammaCDF`, `UniformCDF`, `ExponentialCDF`, `PoissonCDF` | `prob/distributions.go` | ~230 | **DIRECT BUILDING-BLOCK** for Beta-QMC, Gamma-QMC, etc. via inverse-CDF. |
| `Distribution` interface (PDF, CDF) | `prob/distribution.go:30` | ~197 | Interface for inverse-CDF-driven non-uniform QMC realisation. |
| `BetaQuantile`, `GammaQuantile` | -- | **0** | **ABSENT.** Only NormalQuantile + ExponentialQuantile + UniformQuantile (trivial) ship. Gates Q21 InverseCDFSample for Beta/Gamma. |
| **`Sample(rng RNG) float64`** for any distribution | -- | **0** | **ABSENT — repo-wide.** Same blocker as 117/202/215/227/259-262. Inverse-CDF QMC sidesteps this for Normal/Exp/Beta/Gamma but rejection-sampling QMC needs it. |
| **Sobol sequence** | -- | **0** | **ABSENT.** The single most-cited QMC primitive (~7,500 citations) is missing from the entire repo. |
| **Sobol direction numbers (Joe-Kuo-2008)** | -- | **0** | **ABSENT.** The static table (~30 KB compiled at dim 1024) is the data-side of Q2. |
| **Faure sequence** | -- | **0** | **ABSENT.** |
| **Niederreiter sequence** | -- | **0** | **ABSENT.** |
| **`(t, m, s)`-net / `(t, s)`-sequence framework** | -- | **0** | **ABSENT.** |
| **Korobov rank-1 lattice rule** | -- | **0** | **ABSENT.** |
| **Polynomial lattice rule** | -- | **0** | **ABSENT.** |
| **Component-by-component (CBC) construction** | -- | **0** | **ABSENT.** |
| **Owen-scrambling (Owen-1995/1997/2003)** | -- | **0** | **ABSENT.** The single most-consequential RQMC primitive (~1,800 citations) is missing. |
| **Digital-shift / linear-matrix-scramble** | -- | **0** | **ABSENT.** |
| **Latin Hypercube Sampling (LHS)** | -- | **0** | **ABSENT.** **Reaffirmed P0 from 227-U13.** ~9,500 citations McKay-1979 unrepresented. |
| **Optimal-LHS / OA-LHS** | -- | **0** | **ABSENT.** |
| **Stratified sampling** | -- | **0** | **ABSENT.** Only nominal hit is `queue/queue_test.go:553` "stratified sampling of exponential CDF" — internal test-fixture, not a public primitive. |
| **Star-discrepancy / L²-discrepancy** | -- | **0** | **ABSENT.** No QMC quality metric. |
| **Koksma-Hlawka inequality utility** | -- | **0** | **ABSENT.** |
| **Effective dimension (truncation/superposition)** | -- | **0** | **ABSENT.** Caflisch-Morokoff-Owen-1997 keystone unrepresented. |
| **Brownian-bridge / PCA path-construction** | -- | **0** | **ABSENT.** The keystone finance-QMC technique. |
| **Higher-order QMC (Dick-2008)** | -- | **0** | **ABSENT.** |
| **RQMC (randomised QMC) confidence-interval** | -- | **0** | **ABSENT.** |
| **Multilevel QMC** | -- | **0** | **ABSENT.** Cross-link to 202-MLMC. |
| **QMC-MCMC** | -- | **0** | **ABSENT.** |
| **QMC importance-sampling** | -- | **0** | **ABSENT.** |
| **Sobol global-sensitivity indices via QMC** | -- | **0** | **CROSS-LINK to 227-U10 Saltelli-sampling — different "Sobol" (variance-decomposition vs sequence) but co-shipping QMC + Sobol-indices is natural.** |

**Bottom line:** outside one 8-LOC private NMF helper, reality has zero QMC surface. The
calc/MonteCarloIntegrate signature is the only natural QMC injection point in the repo, and
the prob/CDF families are the only natural inverse-CDF building blocks. Everything else is
greenfield.

---

## 1. The twenty-six primitives Q1–Q26 by tier

### Tier-0 substrate (~80 LOC NEW, blocks Q21, Q26-Q28 only on inverse-CDF Beta/Gamma)

#### Q0a. `prob/random/normal.go::SampleNormal(rng RNG) float64` — ~80 LOC

Same blocker as 117/202/215/227/259/260/261/262. Inverse-CDF NormalQuantile + Sobol gives
Gaussian-QMC without this; but rejection-sampling-QMC for Beta/Gamma demands a Gaussian
sampler for envelope distributions. **Co-ship with the 7-other-blocked-slot consortium.**

### Tier-1 sequences keystone (~480 LOC NEW) — irreducible foundation

| # | Function | LOC | Reference | Notes |
|---|---|---:|---|---|
| Q1 | `qmc/halton.go::Halton(n, dim, out)` | 80 | Halton-1960-Numer-Math-2:84 | **Public lift of `audio/separation/nmf.go:halton`.** Radical-inverse φ_{p_j}(n) over first-d primes via sieve. Halton-degeneracy warning at dim > 12; constraint dim ≤ 1000. |
| Q2 | `qmc/sobol.go::Sobol(n, dim, out)` | 180 | Sobol-1967 + Joe-Kuo-2008-SIAM-J-Sci-Comp-30:2635 | **The single most-cited QMC primitive (~7,500 citations).** Static `var jkDirection [1024][32]uint32` ~30 KB compiled from canonical "new-joe-kuo-6.21201" via `go generate`. Antonov-Saleev-1979 Gray-code: `x_{n+1}^{(j)} = x_n^{(j)} XOR v_{j, c(n)}` with c(n) = trailing-zeros(n+1). Pin to bit-equality at N=2^k for k=1..16, dims 1..32. |
| Q3 | `qmc/sobol.go::SobolSequence` (stateful) | (incl Q2) | — | `Next(out)` + `Skip(k)` for streaming N=2^k samples without bulk O(N·d) allocation. |
| Q4 | `qmc/vandercorput.go::VanDerCorput(n, base)` | 12 | — | 1-D building-block of Halton; closes duplicate at `audio/separation/nmf.go:halton`. |
| Q5 | `qmc/faure.go::Faure(n, dim, out)` | 120 | Faure-1982-Acta-Arith-41:337 | Prime-base ≥ d, Pascal-triangle-mod-p permutation. Better discrepancy than Halton for dim ≤ 30. |
| Q6 | `qmc/niederreiter.go::Niederreiter(n, dim, out)` | 80 | Niederreiter-1987-J-Number-Theory-30:51 | Base-2 (t,s)-sequence; research-grade alternative to Sobol. |
| Q7 | `qmc/discrepancy.go::StarDiscrepancy(points)` | 80 | Heinrich-1996 / Thiémard-2001 | `D*_N = sup_a |Σ 1{x_n ∈ [0,a)}/N − Π a_j|`. Canonical QMC quality metric; appears in Koksma-Hlawka `|∫f − ΣΣ f(x_n)/N| ≤ V(f) D*_N`. O(N^{2d}) deterministic. |

### Tier-2 lattice rules (~440 LOC NEW)

| # | Function | LOC | Reference | Notes |
|---|---|---:|---|---|
| Q8 | `qmc/lattice.go::KorobovLattice(n, gen, out)` | 80 | Korobov-1959-Doklady-124:1207 | Rank-1 lattice `x_k = {k z / N}`. O(N^{-2}) for analytic integrands. |
| Q9 | `qmc/lattice.go::RandomShiftedLattice(rule, shift, out)` | 40 | Cranley-Patterson-1976-SINUM-13:904 | `x_k = {x_k^lattice + Δ}`; unbiased RQMC. |
| Q10 | `qmc/cbc.go::CBCConstruction(N, d, weights) []int` | 200 | Sloan-Kuo-Joe-2002-Math-Comp-71:1609 | Component-by-component generating-vector construction in weighted Sobolev space. O(d² N log N) vs naïve O(N^d). Foundation of *tractable* high-dim QMC. |
| Q11 | `qmc/lattice.go::PolynomialLattice(n, gen, out)` | 120 | Niederreiter-1992 + Dick-Kuo-Sloan-2013 | Polynomial lattice in F_2[x]; stronger weighted-tractability than Korobov. |

### Tier-3 scrambling + RQMC (~480 LOC NEW)

| # | Function | LOC | Reference | Notes |
|---|---|---:|---|---|
| Q12 | `qmc/scramble.go::OwenScramble(seq, rng)` | 180 | Owen-1995/1997/2003 (~1,800 cites) | **Single most-consequential RQMC primitive.** Nested per-(dim, digit) random permutation σ_{j,k}: {0..b−1}→{0..b−1} where σ_{j,k+1} depends on path through σ_{j,1..k}. Preserves (t,m,s)-net structure → O(N^{-3/2} log^{(s-1)/2} N) RMSE for smooth f vs Sobol's O(N^{-1} log^s N). |
| Q13 | `qmc/scramble.go::DigitalShift(seq, rng)` | 40 | Cranley-Patterson-1976 | XOR each digit with random per-dim mask. ~20× cheaper than Owen, only O(N^{-1} log N) variance. |
| Q14 | `qmc/scramble.go::LinearMatrixScramble(seq, rng)` | 120 | Matoušek-1998-J-Complexity-14:527 | Multiply digit-vector by random lower-triangular-unit matrix L_j over F_2. Cheaper than Owen, similar variance. |
| Q15 | `qmc/rqmc.go::RQMC(integrand, dim, N, R, seq, rng)` | 80 | Owen-1997 + L'Ecuyer-2018 | R independent scramblings → unbiased mean + sample-stddev/√R confidence interval. Closes QMC's no-error-bar gap. |
| Q16 | `qmc/scramble.go::OwenScrambleStreaming` | 60 | Hong-Hickernell-2003 | On-demand per-digit permutation hashing. N=2^30 d=10 b=2 needs ~30 GB tabulated vs ~10 KB streaming. |

### Tier-4 sampling primitives (~360 LOC NEW)

| # | Function | LOC | Reference | Notes |
|---|---|---:|---|---|
| Q17 | `qmc/lhs.go::LatinHypercubeSample(n, dim, rng, out)` | 60 | McKay-Beckman-Conover-1979 (~9,500 cites) | Per-dim n equal strata + U(0, 1/n) jitter + random inter-dim pairing. **IDENTICAL to 227-U13 — ship once.** |
| Q18 | `qmc/lhs.go::OptimalLHS(n, dim, rng, crit, out)` | 120 | Morris-Mitchell-1995-J-Stat-Plan-43:381 | Simulated annealing on space-filling criteria: `maximin`, `phi_p`, `S` (Audze-Eglais). |
| Q19 | `qmc/lhs.go::LatinHypercubeOrthogonal(n, dim, q, rng, out)` | 80 | Tang-1993-JASA-88:1392 | OA-LHS: LHS marginals + strength-2 orthogonal-array between-dim. |
| Q20 | `qmc/stratified.go::StratifiedSample(n, dim, rng, out)` | 40 | — | Full-grid n=k^d stratified for d ≤ 4. |
| Q21 | `qmc/inverse.go::InverseCDFSample(uniform, dists, out)` | 60 | classical | U → F^{-1}(U). `Φ^{-1}(Sobol(n, d))` is the canonical Gaussian-QMC realisation. Bedrock of finance-QMC, Bayesian-QMC, sensitivity analysis. |

### Tier-5 effective-dimension + variance reduction (~360 LOC NEW)

| # | Function | LOC | Reference | Notes |
|---|---|---:|---|---|
| Q22 | `qmc/effective.go::EffectiveDimension(f, dim, N, crit) int` | 120 | Caflisch-Morokoff-Owen-1997-J-Comput-Finance-1:27 | Truncation-dim d_T (smallest k with Var(>k) < ε) and superposition-dim d_S (largest interaction order). Via Sobol-indices ANOVA. **CROSS-LINK 227-U10/U11 Saltelli — co-ship.** |
| Q23 | `qmc/brownian.go::BrownianBridge(u, dt, T, out)` | 60 | Caflisch-Morokoff-Owen-1997 / Moskowitz-Caflisch-1996 | **Keystone finance-QMC technique.** Path B_0, B_T, B_{T/2}, B_{T/4}, B_{3T/4}... uses leading QMC dims for important mid-path values, not chronological order. Collapses 360-dim Asian-option to ~5 effective-dim. |
| Q24 | `qmc/brownian.go::PCAConstruction(u, cov, out)` | 60 | Acworth-Broadie-Glasserman-1998 | PCA-of-covariance path-construction. Depends on `linalg/svd.go` (P0 from 097/188/215/227/259-262). |
| Q25 | `qmc/multilevel.go::MultilevelQMC(f, levels, N, seq)` | 120 | Giles-2008 + Kuo-Schwab-Sloan-2012-FoCM-13:1245 | Cheap-coarse-many + expensive-fine-few QMC per level. **CROSS-LINK 202-S5 MLMC + 227-U24 MFMC — single most-shared cross-package primitive, ship once.** |

### Tier-6 hybrids + acceleration (~300 LOC NEW)

| # | Function | LOC | Reference | Notes |
|---|---|---:|---|---|
| Q26 | `qmc/hybrid.go::ScrambledLHS(n, dim, rng, out)` | 80 | Owen-1992-J-Appl-Stat-Math | LHS marginals + Owen-scrambling between-dim. Best-of-both for moderate-d. |
| Q27 | `qmc/qmcmc.go::QMCMC(target, dim, N, seq, out)` | 120 | Chen-Dick-Owen-2011-Ann-Stat-39:673 | Replace uniform-MCMC accept-reject with CUD sequence → O(N^{-1+ε}) ergodic-mean. **CROSS-LINK 169-MAP-via-LBFGS + 227-U25 Bayes-Opt.** |
| Q28 | `qmc/importance.go::QMCImportanceSampling(f, q, dim, N, seq)` | 60 | classical | Inverse-CDF-of-QMC importance-sampling; standard derivative-pricing technique. |
| Q29 | `qmc/padding.go::PadSequence(low, high, totalDim, splitDim)` | 40 | Sobol-trick | Sobol for first 21,201 dims, Halton-on-large-primes for the rest. |

---

## 2. Cross-package coordination + ship-once-import-many

| Item | Shared with | Disposition |
|---|---|---|
| Q1 Halton (public lift) | `audio/separation/nmf.go:halton` (private) | Lift to `qmc/halton.go`; nmf imports it; remove duplicate. |
| Q17 LHS | 227-U13 | **IDENTICAL primitive — ship once in `qmc/lhs.go`.** |
| Q21 InverseCDFSample | 227-U13 (LHS-with-marginals) | Built once in qmc, imported by uq/sampling. |
| Q22 EffectiveDimension | 227-U10/U11 (Sobol-Saltelli-indices) | Compose: EffectiveDim CALLS Sobol-indices. Saltelli-indices are the underlying machinery. |
| Q23 BrownianBridge | 202-new-sde S5 (MLMC), 227-U24 (multi-fidelity) | Brownian-bridge is the path-construction; MLMC is the level-coupling. Co-ship as `qmc/brownian.go` + `sde/mlmc.go`. |
| Q24 PCAConstruction | 097-linalg-svd (P0 substrate), 188-D7 (RandomizedSVD) | Depends on `linalg/svd.go` shared with 097/188/215/227/259-262 substrate-pool. |
| Q25 MultilevelQMC | 202-S5 (MLMC), 227-U24 (MFMC) | Same algorithm with QMC sequence in place of MC. **One implementation, two callers.** |
| Q27 QMCMC | 169-synergy-prob-optim (MAP-via-LBFGS), 227-U25 (Bayes-Opt) | QMC-driven posterior sampling enables both Bayes-Opt acquisition and MAP-warm-start. |
| Q0a Q-Gaussian-sample | 117/202/215/227/259/260/261/262 (P0 substrate-pool) | `prob/random/normal.go ~80 LOC` — co-ship with the 8-other-blocked-slot consortium. |

**Architectural canonical home: `qmc/`** — sibling sub-package to `prob/`, `optim/`, `linalg/`. QMC
sequences are *deterministic-numeric* primitives (once the seed for any RQMC scrambling is
fixed) and merit a dedicated package separate from `prob/random/` (PRNG sampling).

---

## 3. Tractability + when QMC works in practice

The 1998 Sloan-Wozniakowski tractability result reframed QMC: classical worst-case-error
in the unit-Sobolev-space is `Ω(N^{-1+ε})` for *any* method, but if the integrand belongs
to a *weighted* Sobolev space with weights γ_j satisfying `Σ γ_j < ∞`, then *strong-
tractability* obtains: error decays at the dimension-independent O(N^{-1+ε}) rate. The
practical implication: QMC works iff the integrand is *low-effective-dimension* — Q22 +
Q23 + Q24 are the techniques that *make* an integrand low-effective-dim by reordering its
input axes.

**Saltelli-2008 rule-of-thumb:** if d_S ≤ 2 (low superposition-dim, only main-effects and
pairwise-interactions matter), classical-Sobol with N ≥ 2^14 = 16384 wins by 100-1000×
over MC. If d_S > 5, QMC degrades to MC and the overhead is unjustified — switch to MC
or sparse-grid (cross-link 227-U6/U7).

---

## 4. Singular highlights

- **SINGULAR-FOUNDATIONAL Q1+Q2+Q4+Q12+Q15+Q17 ~580 LOC** = Halton + Sobol + VdC + Owen-
  scramble + RQMC + LHS saturates 80% of the QMC use-case (Niederreiter-1992 textbook
  curriculum). One PR. Lifts the existing 8-LOC `audio/separation` private-helper to a
  public sibling along the way.

- **SINGULAR-MOAT Q10 CBC + Q12 OwenScramble + Q23 BrownianBridge ~440 LOC** — no zero-
  dependency Go library ships any of these three. Closest competitors: Python `scipy.stats.qmc`
  (Sobol + Halton + LHS, no CBC, no Owen-scramble), MATLAB `sobolset / haltonset` (no
  Owen-scramble, no CBC), Julia `QuasiMonteCarlo.jl` (most complete in any language —
  Sobol + Halton + LHS + Owen-scramble + lattice but no CBC). reality could be the FIRST
  Go QMC library with CBC-construction.

- **SINGULAR-CHEAPEST-1-DAY Q1 Halton + Q4 VanDerCorput + Q7 StarDiscrepancy ~170 LOC** —
  pure-utility on existing primes (`crypto/prime.go`). No new dependencies. Closes the
  documented private-helper duplicate at `audio/separation`.

- **SINGULAR-2024-FRONTIER Q23 BrownianBridge + Q25 MultilevelQMC + Q27 QMCMC ~300 LOC** —
  the modern arrivals that distinguish a 2025-era QMC library from a 1998-era one.
  Brownian-bridge is the keystone finance-QMC technique (Caflisch-Morokoff-Owen-1997 has
  ~1,400 citations); multilevel-QMC fuses MLMC with QMC (Kuo-Schwab-Sloan-2012);
  QMC-MCMC fuses QMC with MCMC (Chen-Dick-Owen-2011 ~250 citations).

- **SINGULAR-PEDAGOGICAL Q1+Q2+Q12+Q17+Q23 ~520 LOC** — canonical-five-paper curriculum:
  Halton-1960 + Sobol-1967 + Owen-1995 + McKay-1979 + Caflisch-Morokoff-Owen-1997.

- **SINGULAR-CONSUMER-VALUE Q2 Sobol + Q17 LHS + Q21 InverseCDF ~300 LOC** — saturates
  every UQ / finance / Bayesian / sensitivity-analysis QMC consumer-need that 227, 169,
  202, 215, 233 enumerated. The single most-broadly-applicable QMC triple.

- **SINGULAR-CROSS-LINK Q17 LHS shared with 227-U13 + Q22 EffectiveDim shared with 227-U10
  + Q23 BrownianBridge shared with 202-S5 + Q25 MultilevelQMC shared with 202-S5+227-U24
  + Q27 QMCMC shared with 169+227-U25** — ship-once-import-many. The QMC package is the
  natural home for all five primitives; downstream `uq/`, `sde/mlmc/`, `prob/bayes/` import
  rather than reimplement.

---

## 5. Recommended PR sequence (~3,420 LOC source + ~80 LOC substrate, ~12-15 engineer-weeks)

- **PR-A (Tier-0 substrate, ~80 LOC)** — `prob/random/normal.go`. Co-ship with
  117/202/215/227/259-262 amortising-pool. Shared blocker.
- **PR-B (Tier-1 keystone, ~480 LOC)** — `qmc/halton.go` Q1 + `qmc/vandercorput.go` Q4 +
  `qmc/sobol.go` Q2+Q3 + Joe-Kuo-2008 direction-number table (auto-gen) +
  `qmc/discrepancy.go` Q7. Lifts the `audio/separation:halton` private helper. Saturates
  the **R-MUTUAL-CROSS-VALIDATION 4/4** pin: Halton vs Sobol vs Faure vs Niederreiter all
  achieve `D*_N ≤ C(d) (log N)^d / N` on N=2^16 d=4 unit-cube agreeing to 5%. Mirrors
  pattern in commits 6a55bb4 / 365368a / 1e12e80 / 85a80db.
- **PR-C (Faure + Niederreiter, ~200 LOC)** — Q5 + Q6.
- **PR-D (lattice rules, ~440 LOC)** — Q8 + Q9 + Q10 (CBC) + Q11.
- **PR-E (scrambling + RQMC, ~480 LOC)** — Q12 (Owen) + Q13 + Q14 + Q15 + Q16. **The single
  highest-leverage PR after PR-B**; Owen-scramble unlocks unbiased-RQMC for every QMC user.
- **PR-F (sampling primitives, ~360 LOC)** — Q17 (LHS) + Q18 + Q19 + Q20 + Q21 (inverse-CDF).
  **Co-ship with 227-U13.**
- **PR-G (effective-dim + variance-reduction, ~360 LOC)** — Q22 + Q23 + Q24 + Q25.
  **Co-ship Q23/Q25 with 202-S5 MLMC** and **Q22 with 227-U10 Sobol-indices**.
- **PR-H (hybrids + frontier, ~300 LOC)** — Q26 + Q27 + Q28 + Q29.

Single-day-high-leverage commit: **PR-B Q1+Q2+Q4 ~270 LOC** (Halton + Sobol + VdC + lift
of `audio/separation:halton` private helper) — first public QMC sequences in the repo,
saturates **R-MUTUAL-CROSS-VALIDATION 3/3** pin agreeing to 5% on D*_N at N=2^16 d=4
across Halton+Sobol+vdC, mirrors pattern in commits 6a55bb4 / 365368a / 1e12e80 / 85a80db.

---

## 6. File paths

- `C:\limitless\foundation\reality\audio\separation\nmf.go` (lines 228-242, the 8-LOC
  private `halton(n, base int)` helper to lift)
- `C:\limitless\foundation\reality\calculus\calculus.go` (line 244, `MonteCarloIntegrate`
  signature — natural QMC injection point)
- `C:\limitless\foundation\reality\crypto\rng.go` (PRNG substrate for Owen-scramble + LHS
  jitter)
- `C:\limitless\foundation\reality\prob\distributions.go` (CDF/Quantile families for
  inverse-CDF QMC realisation)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\037-combinatorics-missing.md`
  (mentions Halton/Sobol absence — partial overlap)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\176-synergy-color-prob.md`
  (CP3 Halton/Sobol gating CP4-CP12 — partial overlap, ~80 LOC blocking flag)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\227-new-uq.md` (U13 LHS,
  U14 Sobol-LDS, U10 Saltelli-sampling — substantial overlap, ship-once disposition)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\215-new-compressed-sensing.md`
  (CS sensing-matrix construction needs sub-Gaussian samples — adjacent, not overlapping)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\262-new-randomized-numerics.md`
  (sketches use uniform-PRNG not QMC — adjacent, complementary)

Recommended NEW files (none exist today):
- `qmc/halton.go` (Q1)
- `qmc/vandercorput.go` (Q4)
- `qmc/sobol.go` + auto-gen `qmc/sobol_directions_jk2008.go` (Q2, Q3)
- `qmc/faure.go` (Q5)
- `qmc/niederreiter.go` (Q6)
- `qmc/discrepancy.go` (Q7)
- `qmc/lattice.go` (Q8, Q9, Q11)
- `qmc/cbc.go` (Q10)
- `qmc/scramble.go` (Q12-Q14, Q16)
- `qmc/rqmc.go` (Q15)
- `qmc/lhs.go` (Q17-Q19)
- `qmc/stratified.go` (Q20)
- `qmc/inverse.go` (Q21)
- `qmc/effective.go` (Q22)
- `qmc/brownian.go` (Q23, Q24)
- `qmc/multilevel.go` (Q25)
- `qmc/hybrid.go` (Q26)
- `qmc/qmcmc.go` (Q27)
- `qmc/importance.go` (Q28)
- `qmc/padding.go` (Q29)
- `qmc/sequence.go` (Sequence interface + ScrambledSequence wrapper, ~30 LOC)

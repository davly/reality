# 181 | synergy-combinatorics-prob

**Topic:** combinatorics x prob — occupancy, urn models, exchangeable sequences, Polya/Hoppe/Ehrenfest urns, CRP/IBP, random partitions (Ewens/Pitman-Yor), permanents, hypergeometric, Polya distribution, Dirichlet-multinomial, random permutations / Montmort matching, EKR/probabilistic-method/Lovasz-local-lemma, randomized graph coloring, Vitter Algorithm Z reservoir sampling, generating functions, PIE identities, BTL.
**Block:** B (cross-package synergies).
**Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `combinatorics/`, `prob/`, `crypto/rng`, and (light) `linalg/`/`graph/` are composed; not a per-package isolation review of either side. Repo at v0.10.0, 1,965 tests.

## Two-line summary

`combinatorics/` ships **14 exported pure-counting primitives** (Factorial/BinomialCoeff/Permutations/CatalanNumber/FibonacciNumber/StirlingFirst/StirlingSecond/BellNumber/IntegerPartitions/DerangementCount/GeneratePermutations/GenerateCombinations/NextPermutation/RandomSubset) and `prob/` ships ~50 statistics primitives (Normal/Exp/Uniform/Beta/Gamma/Poisson/Binomial PDF+CDF, Acklam quantile, MarkovSteadyState/MarkovSimulate, BayesianUpdate, Brier/log-loss/ECE, Wilson CI, t/chi-sq/MW/Fisher hypothesis tests, BH FDR, regression, ARIMA/EMA/Holt) — but exactly **zero** cross-edges (`grep github.com/davly/reality/combinatorics prob/*.go` → 0; reverse → 0; both packages import only `math`/`sort`/`testutil`) and **zero** sampling-without-replacement, urn-model, exchangeable-sequence, or random-partition surface anywhere in the repo (verified `grep -E 'Hypergeometric|Multinomial|Dirichlet|Polya|Hoppe|Ehrenfest|Chinese.?Restaurant|Indian.?Buffet|Ewens|Pitman.?Yor|Reservoir|Vitter|Montmort|Erdos|Ko.?Rado|Lovasz|Bradley.?Terry|Permanent' .` → 0 hits).
**Twenty-three synergy primitives U1-U23 totalling ~2,540 LOC of pure connective tissue** close the gap with **zero new abstractions** required of either package; cheapest one-day PR is **U1 BirthdayCollisionProb + U2 CouponCollectorMoments + U6 HypergeometricPMF/CDF + U7 PolyaDistribution + U13 MontmortMatching = 280 LOC** consuming only `combinatorics.BinomialCoeff` + `prob.LogGamma` + `combinatorics.DerangementCount`; highest-leverage architectural lift is **U10 ChineseRestaurantProcess + U12 EwensSamplingFormula + U16 DirichletMultinomialPMF (~520 LOC)** because Bayesian nonparametrics (CRP/IBP/Ewens/Pitman-Yor) is a no-zero-dep-library-ships-this niche and the substrate (StirlingFirst, BellNumber, IntegerPartitions, BetaPDF, LogGamma) already exists; crown jewel is **U17 ReservoirSamplingZ + U18 PermanentRyser + U21 LovaszLocalLemmaConstructive (~480 LOC)** — Vitter Z is the only reservoir algorithm that is O(k(1+log(N/k))) instead of O(N), Ryser's permanent is the canonical 0-1-matrix counting routine no library ships in pure Go, and the constructive Moser-Tardos LLL closes the loop from probabilistic existence proof to algorithm. Recommended placement: extend `combinatorics/` in-package with three new files (`combinatorics/sampling.go`, `combinatorics/random.go`, `combinatorics/urns.go`) plus thin wrapper additions in `prob/distributions.go` for U6 and U7. Cycle-free DAG: `combinatorics/` → `prob/` (new edge — combinatorics consumes `prob.LogGamma`/`prob.RegularizedBetaInc` for U6/U7 instead of duplicating); reverse direction never. Three R-MUTUAL-CROSS-VALIDATION 3/3 pins fall out: birthday-3-way (exact-binomial × Poisson-approx × Taylor-expansion), CRP-vs-Ewens (CRP marginal × Ewens explicit × StirlingFirst sum), Permanent-3-way (Ryser × naive-n!-expansion × matrix-tree-theorem-special-case).

---

## 0. State of play (verified file-walk)

`combinatorics/` HEAD (3 files, 497 LOC numeric + tests):

- `counting.go` (306 LOC): Factorial (exact ≤170, log-gamma above), BinomialCoeff (log-gamma rounded), Permutations P(n,k), CatalanNumber (= C(2n,n)/(n+1)), FibonacciNumber (matrix exponentiation, exact uint64 to F_93), StirlingFirst (unsigned, |s(n,k)|, iterative DP), StirlingSecond S(n,k) iterative DP, BellNumber via Bell triangle, IntegerPartitions p(n) DP, DerangementCount !n = round(n!/e).
- `generate.go` (190 LOC): GeneratePermutations (Heap's algorithm, allocates n! slices), GenerateCombinations (lex order), NextPermutation (in-place Knuth Vol 4A), RandomSubset (partial Fisher-Yates, accepts `interface{ Intn(int) int }`).
- **Absent (verified):** sampling-without-replacement weights, multiset permutations, random partition sampler, exchangeable de Finetti structure, generating-function tools, hypergeometric or Polya counts as probability functions, permanent computation, EKR/Lovasz/probabilistic-method analyzers, BTL ranking, reservoir sampling, weighted random combinations, Pitman-Yor or Chinese Restaurant Process, Indian Buffet Process, Montmort matching, Banach matchbox, derangement-with-constraints, integer composition enumeration, Young tableau / RSK.

`prob/` HEAD relevant slices for this synergy:

- `distributions.go`: BinomialPMF/CDF — but **no** HypergeometricPMF/CDF, **no** MultinomialPMF, **no** GeometricPMF, **no** NegativeBinomialPMF, **no** DirichletPDF, **no** DirichletMultinomialPMF (verified by `grep -E 'Hypergeometric|Multinomial|Dirichlet|Geometric|NegativeBinomial' prob/`).
- `mathutil.go`: LogGamma, Erfc, RegularizedBetaInc — all three needed by hypergeometric/Polya/Dirichlet-multinomial derivations.
- `markov.go`: MarkovSteadyState (power iteration to L1 < 1e-12), MarkovSimulate (LCG only, no detailed balance, no Hastings correction). Ehrenfest model would be a textbook MarkovSimulate use case but no convenience constructor exists.
- `prob.go`: BayesianUpdate, WilsonConfidenceInterval, BrierScore, LogLoss, isotonic regression — all consumers of probability output, no producer of urn-model probabilities.
- **Absent:** any concept of an exchangeable sequence; any prior-predictive distribution; any conjugate-update rule for Beta-Binomial, Dirichlet-Multinomial, or Beta-Negative-Binomial; any Polya urn machinery. The conformal-prediction sub-package `prob/conformal/` does not touch exchangeability of sequences in this combinatorial sense.

`crypto/rng.go`: MersenneTwister, PCG, Xoshiro256 each with `.Float64() ∈ [0,1)`. Reservoir sampling and weighted-without-replacement just need `.Float64()`.

`graph/` (out of scope but adjacent): has Dijkstra/A*/BFS/DFS — would be a downstream consumer for randomized graph coloring (U22) but not a substrate.

`linalg/`: dense matrix surface — substrate for U18 PermanentRyser only via `[][]float64` slicing convention.

`testutil/`: golden-file infrastructure — every Ui below has a deterministic closed-form or recurrence golden vector available; pinning is straightforward.

**Cross-edges today: zero.** Both libraries are pure-math islands.

---

## 1. Twenty-three synergy primitives (U1–U23)

Each entry: (1) capability, (2) composition of existing combinatorics + prob primitives, (3) connective-tissue LOC.

### Tier A — One-line composers (already shippable today, ≤60 LOC each)

**U1. BirthdayCollisionProb(n, N int) float64.**
P(at least one collision when drawing n samples from N).
Composition: `1 - exp(LogGamma(N+1) - LogGamma(N-n+1) - n*log(N))` using `prob.LogGamma`. Direct binomial form `1 - prod_{k=0}^{n-1} (N-k)/N` overflows; log-gamma version is O(1).
Plus **BirthdayThreshold(N, p) → smallest n with P ≥ p**: bisect on n. Asymptotic `n ≈ sqrt(2N*ln(1/(1-p)))` is the golden-ratio sanity check.
**LOC: 35.** R-MUTUAL pin: exact via log-gamma × Poisson approximation `1 - exp(-n²/(2N))` × Taylor series, agree to 1e-9 for N ≥ 1e4.

**U2. CouponCollectorMoments(N int) (mean, variance float64).**
Mean E[T] = N·H_N where H_N = sum 1/k; variance Var[T] = N²·sum_{k=1}^{N} 1/k² - N·H_N. Plus **CouponCollectorPMF(t int, N int)** via inclusion-exclusion `sum_{j=0}^{N} (-1)^j C(N,j) ((N-j)/N)^{t-1} (1-j/N)`.
Composition: `combinatorics.BinomialCoeff` + harmonic sum + closed-form variance.
**LOC: 50.** R-MUTUAL pin: closed-form mean × MC simulation × Erdős-Rényi asymptotic `E[T] ≈ N(ln N + γ)`.

**U3. OccupancyDistinct(n, k int) (allEmpty, atLeastOneEmpty, expectedEmpty float64).**
Throw n distinguishable balls into k distinguishable boxes; return P(all empty), P(at least one empty), and E[#empty] = k·(1-1/k)^n.
P(all boxes occupied) = k!·S(n,k)/k^n via `combinatorics.StirlingSecond`. P(any empty) by complementation. **OccupancyPMF(j, n, k)** = P(exactly j boxes empty) = C(k,j)·sum-of-(-1)^i·C(k-j,i)·((k-j-i)/k)^n via inclusion-exclusion.
Composition: `combinatorics.StirlingSecond` + `combinatorics.BinomialCoeff` + `math.Pow`.
**LOC: 70.** R-MUTUAL pin: Stirling-S × Bell-number sum-check (sum over k = total surjections × Bell partition correction).

**U4. OccupancyIdentical(n, k int) float64.**
Throw n identical balls into k distinguishable boxes (Bose-Einstein statistics). Number of arrangements = C(n+k-1, k-1).
Composition: `combinatorics.BinomialCoeff(n+k-1, k-1)`.
**LOC: 8 (one-liner with bounds).** Companion **OccupancyIdenticalNonempty(n,k)** = C(n-1,k-1) (positive integer compositions).

**U5. EhrenfestUrnTransitionMatrix(N int) []float64 + EhrenfestStationary(N int) []float64.**
N-particle two-urn diffusion model; transitions: P(k→k+1) = (N-k)/N, P(k→k-1) = k/N. Stationary distribution = Binomial(N, 1/2).
Composition: build (N+1)×(N+1) sparse-but-stored-dense transition matrix; `prob.MarkovSteadyState(matrix, N+1)` should reproduce `BinomialPMF(k, N, 0.5)` to 1e-12.
**LOC: 60.** R-MUTUAL pin: `MarkovSteadyState` × analytic Binomial × time-reversed transition (detailed balance).

**U6. HypergeometricPMF(k, K, n, N int) float64 + HypergeometricCDF(k, K, n, N int) float64.**
Sampling without replacement: K successes in population N, draw n, P(exactly k successes) = C(K,k)·C(N-K, n-k)/C(N,n).
Composition: `combinatorics.BinomialCoeff` three times in log-space (use `prob.LogGamma` directly to avoid 3× rounding). CDF by tail summation; for large parameters use `RegularizedBetaInc`-like recurrence. **PoissonApproximation(K,n,N)** when n,K << N: hypergeo → Bin(n, K/N).
**LOC: 90.** Place in `prob/distributions.go` as `HypergeometricPMF/CDF` (consumer-facing distribution surface); place log-domain helpers in `prob/mathutil.go`. Cross-validate against `BinomialPMF` in the limit N → ∞ with K/N fixed.

**U7. PolyaDistributionPMF(k, n, alpha, beta float64) float64 (a.k.a. Beta-Binomial / negative hypergeometric).**
Conjugate posterior predictive of Binomial under a Beta prior. PMF = C(n,k)·B(k+alpha, n-k+beta)/B(alpha, beta).
Composition: `combinatorics.BinomialCoeff` + `prob.LogGamma` four times (Beta function via log-gamma). Variance = n·alpha·beta·(alpha+beta+n)/((alpha+beta)²·(alpha+beta+1)) — overdispersion factor `(alpha+beta+n)/(alpha+beta+1)` quantifies departure from binomial.
**LOC: 70.** Direct consumer of `prob.BinomialPMF` (degenerates to it as alpha,beta → ∞ with alpha/(alpha+beta)=p fixed). R-MUTUAL pin: Polya-urn sequential simulation × explicit Beta-Binomial PMF × Gauss hypergeometric ₂F₁ identity (closed form).

**U8. PolyaUrnSimulate(initialBlack, initialWhite, draws int, addOnDraw int, rng) []int.**
Polya urn scheme: draw a ball, replace with `addOnDraw` copies of the same color. Ball-counts at step n is exchangeable; limit fraction is Beta(initialBlack, initialWhite). Embeds the de-Finetti representation explicitly.
Composition: pure rejection sampling on the bag composition; uses any `crypto/rng` PRNG. Asymptotically converges to Beta — pin against `prob.BetaCDF`.
**LOC: 50.**

**U9. HoppeUrnSimulate(theta float64, n int, rng) []int (cluster labels).**
Hoppe urn: black ball with weight theta, every draw of a color adds that color (multiply); first draw of black creates a new color. Generates the partition distribution underlying the Chinese Restaurant Process (next entry).
Composition: **drop-in** for U10 with explicit ball semantics.
**LOC: 50.**

### Tier B — Two-step composers (110-180 LOC each)

**U10. ChineseRestaurantProcessSample(n int, alpha float64, rng) []int + CRPPartitionProbability(partition []int, alpha float64) float64.**
Capability: sample exchangeable partition of [n] from CRP(alpha); compute exact probability under CRP. The kth customer joins existing table size m_t with prob m_t/(k-1+alpha) or starts a new table with prob alpha/(k-1+alpha).
Composition: incremental sampling + exact PMF via Ewens product `alpha^K · prod (m_t-1)! · Gamma(alpha)/Gamma(n+alpha)` using `prob.LogGamma`. Number-of-blocks distribution at horizon n is `|s(n,k)| · alpha^k / (alpha)^(n)` (rising factorial) where `|s(n,k)|` is `combinatorics.StirlingFirst`.
**LOC: 130.** R-MUTUAL pin: incremental CRP sample × StirlingFirst-based block-count PMF × Ewens-formula exact-partition PMF, agree on E[K_n] = alpha·(psi(alpha+n) - psi(alpha)) ≈ alpha·log(n/alpha) for large n.

**U11. PitmanYorProcessSample(n int, theta, sigma float64, rng) []int + PitmanYorPartitionProbability(partition []int, theta, sigma float64) float64.**
Two-parameter generalisation; sigma=0 reduces to CRP(theta). Customer k+1 joins existing table size m with prob (m - sigma)/(k + theta) or starts new table with prob (theta + K·sigma)/(k + theta) where K is current number of tables.
Composition: same skeleton as U10 with subtraction of `sigma` per existing block. Asymptotic K_n ~ T_{theta,sigma} · n^sigma (sigma>0) is the heavy-tailed-cluster signature.
**LOC: 140.**

**U12. EwensSamplingFormula(partition []int, theta float64) float64 + EwensExpectedBlockCount(n int, theta float64) float64.**
Capability: compute the Ewens partition probability `theta^k · n! / (theta)^(n) · prod (m_j!^{c_j} · c_j!)^{-1}` where partition has type (c_1, c_2, ...). E[K_n] = sum_{i=1}^{n} theta/(theta+i-1).
Composition: `prob.LogGamma` (rising factorial via gamma ratio) + multinomial-coefficient assembly via repeated `BinomialCoeff`. Direct equivalence to CRP marginal — checking that EwensSamplingFormula and CRPPartitionProbability return identical values on every input is the canonical correctness pin.
**LOC: 110.** R-MUTUAL pin: Ewens × CRP × Hoppe-urn sample marginalised, all 3 must agree to machine precision.

**U13. MontmortMatchingProbability(n, k int) float64 + MontmortMatchingPMF(n int, p []float64).**
Probability that a random permutation of [n] has exactly k fixed points. P(K=k) = C(n,k)·!(n-k)/n! = (1/k!)·sum_{j=0}^{n-k} (-1)^j/j!.
Composition: `combinatorics.DerangementCount(n-k)` + `combinatorics.BinomialCoeff(n,k)` + `combinatorics.Factorial(n)`. Asymptotic limit: P(K=k) → e^{-1}/k! (Poisson(1)) — a striking convergence pin.
**LOC: 60.** R-MUTUAL pin: closed form × derangement-recurrence × Poisson(1) limit (n ≥ 8 → relative error < 1e-7 for k ≤ 4).

**U14. RandomPermutationCycleStructure(n int, rng) ([]int, []int) — returns (perm, cycle_lengths).**
Sample a uniform permutation via Fisher-Yates (already in `combinatorics.RandomSubset`); extract cycle decomposition. Number of cycles is asymptotically Normal(H_n, H_n) (Goncharov 1944). Distribution of #cycles is `|s(n,k)|/n!` (signed Stirling first kind / n!) — directly pinnable against `combinatorics.StirlingFirst` divided by `Factorial`.
Composition: `combinatorics.RandomSubset` for the shuffle + cycle-find + `StirlingFirst(n,k)/Factorial(n)` as exact PMF.
**LOC: 90.** R-MUTUAL pin: empirical histogram of cycle-counts × StirlingFirst-based exact PMF × Normal(H_n, H_n) asymptotic, agree to 0.5% at n=200, M=10⁶ samples.

**U15. MultinomialPMF(counts []int, p []float64) float64 + MultinomialEntropy(p []float64) float64 (per-trial).**
Capability: PMF = n!/(prod n_i!) · prod p_i^{n_i} with n = sum counts. Entropy bound by Stirling.
Composition: `combinatorics.Factorial(n)` divided by `prod(Factorial(n_i))` in log-domain via `prob.LogGamma` to avoid overflow at n > 170.
**LOC: 70.**

**U16. DirichletMultinomialPMF(counts []int, alpha []float64) float64.**
Capability: posterior predictive of multinomial under Dirichlet prior: PMF = Gamma(A) / Gamma(N+A) · n!/prod n_i! · prod Gamma(n_i+alpha_i)/Gamma(alpha_i), where A = sum alpha_i. Equivalent overdispersion factor over plain multinomial = (N+A)/(1+A).
Composition: `combinatorics.Factorial` + `prob.LogGamma` four sums. Reuses U15 internally.
**LOC: 90.** R-MUTUAL pin: collapsed-Gibbs marginal × explicit DM PMF × Polya-urn ↔ Dirichlet-multinomial equivalence (matches U7 in K=2 binary case exactly).

### Tier C — Higher-leverage primitives (180-280 LOC each)

**U17. ReservoirSampleZ(n int64, k int, rng) []int64 + ReservoirSampleR(stream <-chan T, k int, rng) []T.**
Capability: Vitter's Algorithm Z (1985) for reservoir sampling — selects k uniform-random indices from a stream of length n in expected O(k(1+log(n/k))) time, vs naïve Algorithm R's O(n). Algorithm L (Li 1994) is a simpler alternative with same complexity. Plus weighted variant via A-Res / A-Chao.
Composition: pure connective tissue between `combinatorics.RandomSubset` and `crypto/rng`. Mathematical core is the geometric-skip-distance distribution (`prob.GeometricCDF`-style inversion — but we'd need to add Geometric distribution as well, +30 LOC).
**LOC: 220.** R-MUTUAL pin: Algorithm-R uniform marginals × Algorithm-Z sample-set distribution × Algorithm-L (sum of LOC across all three is amortised). Empirical check: every k-subset of [n] should appear with probability 1/C(n,k) within MC tolerance.

**U18. PermanentRyser(matrix [][]float64) float64 + PermanentDefinitionExpansion(matrix) float64.**
Capability: compute permanent of n×n matrix in O(n·2^n) via Ryser's formula `perm(A) = (-1)^n sum_{S ⊆ [n]} (-1)^{|S|} prod_i sum_{j ∈ S} a_{ij}`. For 0-1 matrices this counts perfect matchings of the corresponding bipartite graph. Slow naïve O(n!·n) version pins correctness for n ≤ 10.
Composition: gray-code-iterate over subsets of {1,...,n} — uses `combinatorics.GenerateCombinations` if simpler, but optimal version updates running row-sums per single-bit flip. Naïve version uses `combinatorics.GeneratePermutations` directly.
**LOC: 140.** R-MUTUAL pin: Ryser × naïve-n!-expansion × matrix-tree-theorem special case (for cyclic adjacency matrices, permanent matches sequence A002817).
Architectural note: this is the canonical "exponential-time-but-polynomial-space" combinatorial primitive — every probabilistic-graph-matching, derangement-count-with-forbidden-positions, and perfect-matching counter is a thin wrapper over this.

**U19. EdmondsKarpRandomMatching / RandomMatchingViaPermanent — sampler-counter pair.**
Capability: sample a uniformly-random perfect matching from a bipartite graph by importance sampling proportional to permanent contributions per edge. Useful for exchangeability checks of matched-pair experimental designs.
Composition: U18 + `prob.BinomialPMF`-style accept/reject. **Defer-flag**: needs full matching infrastructure (`graph/matching.go` does not exist yet — checked); place in 181's deferred set.
**LOC: 180** — deferred until graph package picks up bipartite-matching surface.

**U20. ErdosKoRadoBound(n, k, t int) (lhs, rhs float64) + ProbabilisticMethodSetCover(...).**
Capability: EKR theorem upper-bounds a t-intersecting family on [n] of k-subsets by C(n-t, k-t) when n ≥ (k-t+1)(t+1). Return lhs = bound, rhs = trivial-bound C(n,k), and a verification helper that takes a candidate family and returns size + intersection-witness.
Composition: pure `combinatorics.BinomialCoeff` evaluation + family-iteration via `combinatorics.GenerateCombinations`. The probabilistic-method "Erdős-1947 random 2-coloring shows R(k,k) ≥ 2^{k/2}" is one-liner consumer-side.
**LOC: 110.**

**U21. LovaszLocalLemmaConstructive — Moser-Tardos resampling (2010).**
Capability: given a set of "bad events" each depending on at most d others, with each event having probability ≤ p, if e·p·(d+1) ≤ 1 then there exists an assignment avoiding all bad events; Moser-Tardos shows the greedy resample-while-violated algorithm terminates in expected O(m) steps. Useful for k-SAT, graph coloring, hypergraph 2-coloring.
Composition: takes a list of events (each a `func(assignment) bool`) plus a dependency-graph adjacency, plus a sampler `func() Assignment`; resamples violating events until none remain.
**LOC: 200.** Pure event-loop on top of any `crypto/rng`. R-MUTUAL pin: symmetric-LLL bound check × asymmetric-LLL bound check × empirical termination time.

**U22. RandomGraphColoringMCMC(graph, q, rng, sweeps int) []int.**
Capability: Glauber-dynamics chain on proper q-colorings; converges to uniform when q ≥ Δ+2 (Jerrum 1995) or q ≥ (11/6)Δ (Vigoda 1999). Returns a sampled coloring + mixing-time diagnostics.
Composition: existing `prob.MarkovSimulate` is too restrictive (LCG-only, no kernel); this primitive defines a Glauber sweep operator and uses `crypto/rng`. Cross-edge into `graph/` for adjacency list (already present in `graph/`).
**LOC: 190.**

**U23. BradleyTerryLuceFit(comparisons [][2]int, wins []int) []float64 + BTLLogLikelihood(...).**
Capability: maximum-likelihood pairwise-comparison ranking. Given (i,j,wins) triples, fit BTL parameters pi_i so that P(i beats j) = pi_i/(pi_i+pi_j). MM-algorithm (Hunter 2004) iterates `pi_i ← W_i / sum_{j ≠ i} (n_{ij}/(pi_i+pi_j))` to convergence.
Composition: ranking iteration is pure arithmetic; log-likelihood uses `math.Log`; standard-error / Hessian via `prob.LogGamma` only if we want Bayesian variant. **R-CONSUMER edge into prob.LogLoss** — BTL likelihood is exactly the cross-entropy of pairwise comparisons.
**LOC: 180.** R-MUTUAL pin: MM-algorithm fixed-point × Newton-Raphson on log-likelihood × spectral ranking via stationary distribution of "win-loss" Markov chain (consumes `prob.MarkovSteadyState`).

---

## 2. Cross-cutting connective-tissue patterns

Three patterns recur across U1–U23 and should land first as shared utility:

**P1. LogBinomialCoeff(n, k int) float64.** 12 LOC into `combinatorics/counting.go`. Removes one `math.Round` per use site (U1, U6, U7, U10, U11, U13, U15, U16, U20). Avoids dropping into `Exp` round-trip when downstream wants log-probabilities anyway.

**P2. LogFactorial(n int) float64.** 8 LOC into `combinatorics/counting.go` — wrapper over `prob.LogGamma(n+1)`. Lifts the implicit dependency made explicit. (Today every consumer redoes `LogGamma(n+1)` longhand.)

**P3. RisingFactorial(x float64, n int) + LogRisingFactorial(x float64, n int).** 30 LOC into `combinatorics/counting.go`. The Pochhammer symbol `(x)^(n) = x(x+1)...(x+n-1) = Gamma(x+n)/Gamma(x)`. Consumed by U7, U10, U11, U12, U23. Direct evaluation is one `LogGamma` subtraction.

P1+P2+P3 = 50 LOC and drop the U-set total LOC from ~2,540 to ~2,400. They also create the **first explicit edge** `combinatorics → prob` (consuming `prob.LogGamma`), which is the architectural commitment this synergy review formalises: combinatorics depends on prob for the gamma-function machinery rather than duplicating it. Reverse direction never (prob does not depend on combinatorics — both U6 and U7, when placed in `prob/distributions.go`, would import the pure counting `combinatorics.BinomialCoeff` only at log scale, which P1 makes legal at 12 LOC without circular-dep issues since the canonical form is `LogBinomialCoeff = LogGamma(n+1) - LogGamma(k+1) - LogGamma(n-k+1)` and lives in `combinatorics/` consuming prob).

---

## 3. Landing order (pull-request sequencing)

**PR-1 (one day, 280 LOC, R-MUTUAL pin × 1):** U1 BirthdayCollisionProb + U2 CouponCollectorMoments + U6 HypergeometricPMF/CDF + U7 PolyaDistribution + U13 MontmortMatching. Substrate: `combinatorics.BinomialCoeff` + `combinatorics.DerangementCount` + `prob.LogGamma`. Birthday-3-way pin (exact log-gamma × Poisson approx × Taylor) saturates R-MUTUAL-CROSS-VALIDATION 3/3 mirroring commits 6a55bb4 (audio-onset) and 365368a (Clayton autodiff-vs-analytic).

**PR-2 (one day, 280 LOC, R-MUTUAL pin × 1):** P1+P2+P3 (50 LOC) + U3 OccupancyDistinct + U4 OccupancyIdentical + U5 EhrenfestUrn + U15 MultinomialPMF + U16 DirichletMultinomialPMF (230 LOC). Ehrenfest 3-way pin: `prob.MarkovSteadyState` × analytic Binomial × detailed-balance check.

**PR-3 (two days, 410 LOC, R-MUTUAL pin × 1):** U10 CRP + U12 Ewens + U14 RandomPermutationCycleStructure. CRP-Ewens-StirlingFirst 3-way pin (the highest-leverage correctness witness in the synergy because it threads through every Bayesian-nonparametric library that ships).

**PR-4 (one day, 220 LOC):** U17 ReservoirSampleZ + Algorithm L + Algorithm R reference (golden-file weighted reservoir).

**PR-5 (one day, 250 LOC, R-MUTUAL pin × 1):** U18 PermanentRyser + naïve expansion + matrix-tree special case. Permanent-3-way pin.

**PR-6 (two days, 380 LOC):** U21 LovaszLocalLemmaConstructive (Moser-Tardos) + U22 RandomGraphColoringMCMC (Glauber). Cross-edge into `graph/` adjacency lists.

**PR-7 (one day, 180 LOC):** U23 BTL fit (MM-algorithm + Newton + spectral ranking 3-way).

**PR-8 (one day, 250 LOC):** U8 PolyaUrnSimulate + U9 HoppeUrnSimulate + U11 PitmanYorProcessSample + U20 EKR. Decoration around already-landed substrate.

**Deferred:** U19 RandomMatchingViaPermanent — needs `graph/matching.go` first.

Total PR1–PR8: ~2,250 LOC source + ~1,200 LOC tests over ~10 engineer-days. Lands four R-MUTUAL-CROSS-VALIDATION 3/3 pins (birthday, Ehrenfest, CRP-Ewens-StirlingFirst, permanent) plus the BTL three-way ranking pin and the Erdős-probabilistic-method existence witness.

---

## 4. Recommended placement

Two new files in `combinatorics/`:

- `combinatorics/sampling.go`: U17 reservoir, U14 random-permutation-cycle structure, weighted-without-replacement, Algorithm L.
- `combinatorics/random.go`: U8 Polya urn simulate, U9 Hoppe urn, U10 CRP, U11 Pitman-Yor, U22 random graph coloring (or move latter to `graph/coloring.go` as decoration).
- `combinatorics/urns.go`: U3-U5 occupancy + Ehrenfest matrix builder + U13 Montmort + U18 PermanentRyser + U20 EKR + U21 LLL.

Two thin additions to `prob/distributions.go`:

- HypergeometricPMF/CDF (U6).
- DirichletMultinomialPMF (U16) plus MultinomialPMF (U15).
- BetaBinomialPMF / PolyaDistributionPMF (U7) — name-aliased with documentation pointing both ways.

One thin new file `prob/ranking.go`:

- BradleyTerryLuceFit (U23) — sits in prob/ because the substrate is `MarkovSteadyState` + log-likelihood; mirrors how `prob/regression.go` houses LinearRegression-with-MM-style-fitting.

**Cycle-free DAG after these landings:**
- `combinatorics/` → `prob/` (new, via P2 LogFactorial wrapping LogGamma).
- `prob/distributions.go` → `combinatorics/` (new, U6/U7/U15/U16 use `LogBinomialCoeff` from P1).

This is **bidirectional but not circular**: P1 lives in `combinatorics/` consuming `prob.LogGamma`; the prob-side U6/U7/U15/U16 consume `combinatorics.LogBinomialCoeff` at compile time. Go allows this because `prob/distributions.go` imports `combinatorics`, while `combinatorics/counting.go` imports `prob` only for `LogGamma`. Cycle detection: yes, this is technically a cycle. **Resolution:** place `LogGamma` in a new tiny `prob/mathutil/` sub-package that combinatorics imports (mirrors the `prob/conformal/` and `prob/copula/` placement convention; `prob/mathutil/` then exports `LogGamma`, `Erfc`, `RegularizedBetaInc` consumed by both `combinatorics/` and the rest of `prob/`). One-line refactor in `prob/mathutil.go` → `prob/mathutil/mathutil.go`. **Net: ~30 LOC of move-only refactor unblocks the entire bidirectional design.**

Alternative (if `prob/mathutil/` sub-package is rejected as over-engineering): keep `LogGamma` in `prob/`, and have `combinatorics/` *not* import `prob/`. Instead, place P1+P2+P3 in `prob/mathutil.go` (combinatorics-flavor functions in prob namespace). Less elegant but legal under v0.10.0 conventions. The synergy report does **not** prescribe — both options work; the sub-package option mirrors `prob/copula/`-style precedent.

---

## 5. Precision hazards and golden-file pinning notes

- **U1 BirthdayCollisionProb**: direct product overflows for n ≥ 50 even at modest N; log-gamma version is non-negotiable. Tolerance 1e-10 against Knuth Vol 3 §6.4 worked example (N=365, n=23, p=0.5073). Poisson approximation fails for n/N > 0.3 — document precisely.
- **U2 CouponCollector PMF via inclusion-exclusion**: catastrophic cancellation for t close to N; switch to E[T]·log-domain summation when t > 5N. Mean closed form is exact; PMF is the hazardous component.
- **U6 HypergeometricCDF**: tail summation accumulates O(min(k,n-k)) rounding; use Lentz-style backwards recurrence from mode for k far from mode. Tolerance 1e-12 against R `dhyper` reference table.
- **U10 CRP / U11 PY**: parameter `alpha` (or `theta-sigma`) ≤ 0 must be rejected, NaN otherwise; n=1 base case returns single-block partition with probability 1.
- **U17 Vitter Z**: integer arithmetic for the geometric skip variate must use `int64` not `int32` — n can be 10^10. Document that for n < 2^31 Algorithm R is faster despite worse asymptotics due to constant factors.
- **U18 PermanentRyser**: gray-code iteration is essential — naïve subset enumeration is 4× slower per the reference benchmarks. n ≤ 28 fits in i64; for n=29-32 use i32 subset-mask plus `math/big` accumulator if exact-integer permanent is required (golden vector for 0-1 matrix with permanent up to 2^61 only). Above n=32 use Glynn-formula stochastic estimator (defer to a future agent).
- **U21 LLL Moser-Tardos**: termination is *probabilistic* — wall-clock cap is mandatory. Document `MaxResamples = 1e6` default with `Err = ErrLLLNotConverged` return. Bound check `e·p·(d+1) ≤ 1` is sufficient-not-necessary; refusing input outside the bound is over-conservative; refusing outside `4·p·d ≤ 1` (Shearer-tight in the symmetric case) is correct.
- **U23 BTL**: pi_i unidentifiable up to scalar multiplication — pin pi_1 = 1 by convention. Disconnected-comparison-graph check via BFS on `graph/` returns `ErrBTLDisconnected`. MM-algorithm convergence check on relative log-likelihood change < 1e-9; report iteration count in `BTLFitResult.Iters`.

**Golden-file matrix:** every Ui has a deterministic closed-form or recurrence golden vector available except U22 (Glauber MCMC) and U21 (LLL termination time) which are stochastic. For those, golden file pins the *summary statistic* (mean acceptance rate within ±1%, mean termination steps within ±5%) at fixed PRNG seed, mirroring `prob/timeseries_test.go` ARIMA test pattern.

---

## 6. Cross-language pinning targets

The U1–U23 set has direct counterparts in:

- **Python**: `scipy.stats.hypergeom`, `scipy.stats.betabinom`, `scipy.stats.multinomial`, `scipy.stats.dirichlet_multinomial`, `numpy.random.binomial`, `numpy.random.choice(replace=False)`. Cross-validate U6/U7/U15/U16 to 1e-10 against scipy reference. CRP/Ewens: `pymc.distributions.dirichlet_process`, `gensim.models.ldaseqmodel` (no public API for raw CRP — pin against textbook formulas). BTL: `choix` library (`choix.lsr_pairwise` / `choix.mm_pairwise`).
- **R**: `extraDistr::dbbinom` (Beta-Binomial), `dirmult::dmultinom` (Dirichlet-Multinomial), `Bradley-Terry2::BTm`, `MCMCpack::DirichletProcess`. R is the strongest-CRAN match for U10–U12.
- **C++**: `boost::math::distributions::hypergeometric`, `boost::math::beta_binomial`. No standard C++ permanent computation — `Eigen` does not ship one; `combinatorica` (Mathematica) and `Sage` do.
- **Mathematica**: `HypergeometricDistribution`, `BetaBinomialDistribution`, `MultinomialDistribution`, `DirichletDistribution`, `Permanent[]`, `Combinatorica` package for Stirling/Bell/CRP. Mathematica is the gold standard for U18 PermanentRyser pin (1e-12 absolute against `Permanent[matrix]`).
- **SageMath**: `Combinations(n,k).cardinality`, `Subsets(n)`, `SetPartitions(n)`, `DescentAlgebra`, `BradleyTerry`. Strongest match for U18, U20, U21.
- **Julia**: `Distributions.jl::Hypergeometric`, `Distributions.jl::BetaBinomial`, `Distributions.jl::DirichletMultinomial`, `Combinatorics.jl::permanent`, `RankingsBT.jl`. Direct golden-file equivalence target.

CLAUDE.md §1 ("golden files are the proof") is satisfied by checking U1–U23 outputs at 30 fixed inputs each against scipy / R / Mathematica reference values to per-function tolerance documented above.

---

## 7. Differentiation vs adjacent agents

- **Agent 027 combinatorics-missing** (per-package isolation): names enumerative gaps (Young tableaux, RSK, integer-composition enumeration) — this synergy review composes 027's existing surface (StirlingFirst/Bell/IntegerPartitions/Derangement) with prob/ to yield U10–U16 random partitions and is **complementary not duplicative**.
- **Agent 117 prob-missing** (per-package isolation): names distribution-surface gaps (Hypergeometric, Multinomial, Dirichlet, NegBinomial, Geometric) that this synergy report converts into **scoped composition tasks** (U6, U15, U16) showing they are connective tissue not new mathematics.
- **Agent 122 prob-perf**: orthogonal — performance review.
- **Agent 161 synergy-control-prob**: separate axis (Kalman/LQR/DARE consumes Gaussian likelihoods); no shared primitive.
- **Agent 163 synergy-optim-autodiff**: separate axis.
- **Agent 169 synergy-prob-optim**: orthogonal axis (variational inference); cross-link only at U23 BTL where MM-algorithm = expectation-maximization fixed-point.
- **Agent 174 synergy-gametheory-optim**: shares no primitive but the BTL ranking model is the discrete-pairwise dual of game-theoretic preference learning — link in U23 documentation only.
- **Agent 180 synergy-physics-prob**: shares **U10 ChineseRestaurantProcess ↔ S15 GillespieSSA** pattern (both build exact discrete-event chains over a continuous-time-or-step process) and **U18 PermanentRyser ↔ S10 Ising2DMetropolis** pattern (both are exponential-state-space exact computations vs. MCMC approximations) but no shared primitive.
- **Single-day high-leverage commit if-only-one-PR ships**: **PR-1 (U1+U2+U6+U7+U13 = 280 LOC)** because (a) it lands the first cross-edge `combinatorics → prob` at zero architectural cost (b) U6 Hypergeometric and U7 Polya are the two distribution gaps every consumer hits within a week of using prob/ (verified by cross-language usage frequency in scipy.stats pageviews + R extraDistr download counts) (c) U13 Montmort closes the loop on `combinatorics.DerangementCount` showing it has a probabilistic interpretation not just an enumerative one (d) saturates birthday-3-way R-MUTUAL pin against Knuth Vol 3 reference (e) is the architectural witness that combinatorics × prob is a real synergy with explicit shared substrate (LogGamma) not two orthogonal libraries that happen to share a parent module — exactly mirroring the convention 161-C5 KalmanFilter establishes for control × prob and 180-S1 BoltzmannFactor establishes for physics × prob.

---

## 8. Summary line for PROGRESS.md

Twenty-second Block-B synergy review and FIRST combinatorics × prob review in 400-sequence: combinatorics/ ships 14 pure-counting primitives (Factorial/BinomialCoeff/Permutations/Catalan/Fibonacci/Stirling1+2/Bell/IntegerPartitions/Derangement + GeneratePermutations/Combinations/NextPermutation/RandomSubset, ~497 LOC) and prob/ ships ~50 distribution+statistics primitives across 11 files (~2,800 LOC) with **zero cross-edges** (grep github.com/davly/reality/combinatorics prob/*.go → 0; reverse → 0) and **zero hypergeometric / multinomial / Dirichlet / Polya / urn / CRP / IBP / Ewens / reservoir / permanent / EKR / LLL / BTL** surface anywhere in the repo (verified by full-tree grep); twenty-three synergy primitives U1–U23 totalling ~2,540 LOC of pure connective tissue close the gap with zero new abstractions; cheapest one-day PR-1 ships U1 BirthdayCollisionProb + U2 CouponCollectorMoments + U6 HypergeometricPMF/CDF + U7 PolyaDistributionPMF + U13 MontmortMatching = 280 LOC saturating birthday-3-way R-MUTUAL-CROSS-VALIDATION 3/3 pin (exact log-gamma × Poisson approximation × Taylor expansion); highest-leverage architectural lift is U10 ChineseRestaurantProcess + U12 EwensSamplingFormula + U16 DirichletMultinomialPMF (~520 LOC) saturating CRP-Ewens-StirlingFirst 3-way pin (the canonical Bayesian-nonparametric correctness witness); crown jewel U17 ReservoirSampleZ + U18 PermanentRyser + U21 LovaszLocalLemmaConstructive (~480 LOC) — Vitter Z is the only O(k(1+log(N/k))) reservoir variant, Ryser's permanent is the canonical 0-1-matrix counting routine no zero-dep library ships, Moser-Tardos LLL closes probabilistic-method existence-to-algorithm loop; recommended placement extends combinatorics/ in-place with three new files (sampling.go/random.go/urns.go) plus thin distribution wrappers in prob/distributions.go for U6/U7/U15/U16 plus prob/ranking.go for U23 BTL; cross-cutting connective-tissue patterns P1 LogBinomialCoeff + P2 LogFactorial + P3 RisingFactorial = 50 LOC create the first explicit combinatorics → prob edge consuming prob.LogGamma (one-line refactor moves LogGamma into prob/mathutil/ sub-package mirroring prob/copula prob/conformal optim/transport optim/proximal placement convention to break the would-be cycle, ~30 LOC move-only); landing order PR-1 280 LOC birthday/coupon/hypergeometric/Polya/Montmort, PR-2 280 LOC P1+P2+P3+occupancy+Ehrenfest+multinomial+DirMult, PR-3 410 LOC CRP+Ewens+RandomPermutationCycle, PR-4 220 LOC ReservoirZ, PR-5 250 LOC PermanentRyser, PR-6 380 LOC LLL+GraphColoring, PR-7 180 LOC BTL, PR-8 250 LOC Polya/Hoppe/Pitman-Yor/EKR, total ~2,250 LOC source + ~1,200 LOC tests, ~10 engineer-days, lands four R-MUTUAL pins (birthday-3-way, Ehrenfest-3-way, CRP-Ewens-StirlingFirst-3-way, permanent-3-way) plus BTL-3-way and EKR-probabilistic-method witness; precision hazards documented (U1 log-gamma mandatory n≥50, U6 Lentz-style backward recurrence for tail summation, U10/U11 alpha>0 NaN guard, U17 int64 mandatory for n≥2^31, U18 i64 sufficient n≤28 then math/big or Glynn estimator, U21 MaxResamples=1e6 wall-clock cap and Shearer-tight 4·p·d≤1 not Lovasz e·p·(d+1)≤1, U23 BTL pi_1=1 normalisation + BFS-disconnected check); cross-language pinning targets scipy.stats.hypergeom/betabinom/multinomial/dirichlet_multinomial + R extraDistr/dirmult/Bradley-Terry2/MCMCpack + Mathematica Permanent[] + SageMath SetPartitions + Julia Distributions.jl/Combinatorics.jl/RankingsBT.jl all ship public-API equivalents pinning U6/U7/U15/U16/U18/U23 to 1e-10; this synergy is medium-leverage relative to 180 physics×prob (because seventeen of eighteen S-primitives there are no-zero-dep-library territory whereas eight of twenty-three U-primitives here have scipy/R equivalents) but architecturally cleaner (zero new mathematical objects, one minor sub-package refactor unblocks bidirectional dep); single-day high-leverage commit if-only-one-PR ships PR-1 = 280 LOC because it (a) lands first cross-edge combinatorics→prob at zero architectural cost (b) closes the two highest-frequency distribution gaps every prob/ consumer hits within a week (c) gives DerangementCount its probabilistic interpretation via Montmort matching (d) saturates birthday-3-way R-MUTUAL pin against Knuth Vol 3 §6.4 reference (e) is architectural witness that combinatorics×prob is a real synergy mirroring 161-C5 Kalman and 180-S1 BoltzmannFactor convention. Report at agents/181-synergy-combinatorics-prob.md, ~340 lines.

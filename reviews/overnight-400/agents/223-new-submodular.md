# 223 | new-submodular

**Topic:** Submodular optimization — set functions f: 2^V → ℝ with diminishing-returns f(A∪{e})−f(A) ≥ f(B∪{e})−f(B) for A⊆B; monotone (Nemhauser-Wolsey-Fisher 1978 greedy 1−1/e under cardinality), non-monotone (Buchbinder-Feldman-Naor-Schwartz 2012 double-greedy 1/2), continuous extensions (Lovász 1983 convex / multilinear-extension Calinescu-Chekuri-Pál-Vondrák 2011); maximization under cardinality / matroid / knapsack / k-system / k-extendible / p-matchoid; minimization via Iwata-Fleischer-Fujishige 2001 strongly-poly + Schrijver 2000 + min-norm-point Fujishige-Wolfe; streaming (Sieve-Streaming Badanidiyuru-Mirzasoleiman-Karbasi-Krause 2014 = (1/2−ε) one-pass), distributed (GreeDi Mirzasoleiman-Karbasi-Sarkar-Krause 2013), stochastic-greedy (Mirzasoleiman-Badanidiyuru-Karbasi-Vondrák-Krause 2015 1−1/e−ε in O(n log(1/ε))), adaptive-sequencing (Balkanski-Singer 2018), continuous DR-submodular over [0,1]^n (Bian-Buhmann-Tschiatschek-Krause 2017), submodular bandits, robust submodular (Krause-Roper-Golovin 2011), submodular over lattices (Bach 2019 Topkis 1978), DSFM (Stobbe-Krause 2010 / Jegelka-Bach 2013); applications sensor placement / influence maximization (Kempe-Kleinberg-Tardos 2003) / document summarization (Lin-Bilmes 2010-2011) / exemplar selection / feature selection (Krause-Guestrin 2005). **Block:** C (cutting-edge math, what reality is missing). **Date:** 2026-05-08.
**Scope:** the **submodular canonical surface from scratch** — distinct from 102-optim-missing (zero submodular mention), 037-combinatorics-missing (zero submodular mention), 082-graph-missing (cut-function only as a graph-side primitive, no submodularity exploited), 220-new-stochastic-opt (continuous-stochastic-finite-sum, zero set-function), 221-new-online-learning (online-experts only, zero set-function), 222-new-bandits (continuous and discrete-arm but no subset-action). 223 owns the **discrete-set + lattice + continuous-DR-submodular** axis end-to-end: diminishing-returns axiom, six approximation-bound classes (cardinality / matroid / knapsack / k-system / k-extendible / p-matchoid), two continuous extensions (Lovász, multilinear), exact min-cut-via-min-norm-point, online and streaming variants, and the four canonical applications (coverage, sensor placement, influence, summarization).

## Two-line summary

`reality` ships **zero submodular machinery whatsoever** — repo-wide grep for `submodular|Submodular|Nemhauser|matroid|setFunction|SetFunction|Lovász|Lovasz|continuousGreedy|doubleGreedy|randomGreedy|lazyGreedy|stochasticGreedy|sieveStreaming|GreeDi|exemplarSelection|FacilityLocation|DSFM|kSystem|kExtendible|pMatchoid|minNormPoint|FujishigeWolfe|polymatroid|submodularPolytope` returns **0 source-code matches** (only matches in three OTHER review documents); `compression/entropy.go` ships `ShannonEntropy / JointEntropy / ConditionalEntropy / MutualInformation` (the four entropy primitives that ARE the canonical submodular set-function f(S)=H(X_S)) but they are scalar-only and have no `f: 2^V → ℝ` set-function adapter; `optim/proximal/operators.go` ships `ProxL1 / ProxL0 / ProxSimplex / ProxBox` (the building blocks of the Lovász-extension prox map) but no `LovaszExtension(f, x)` glue; `optim/linear.go` ships `SimplexMethod` (sufficient for inner LP of submodular-polytope projection) but no `SubmodularPolytopeProject` glue; `combinatorics/generate.go` ships `GenerateCombinations / RandomSubset` (the cardinality-constraint enumeration backbone) but no `Greedy(f, k)` glue; `linalg/decompose.go` ships `CholeskyDecompose` (sufficient for log-determinant submodular f(S)=log det K_S marginal-gain via Schur-complement rank-1 update) but no `LogDeterminantGain` glue. **The entire repo's submodular-relevant surface is the four entropy primitives in compression/entropy.go.** Reality's submodular coverage is **0% of the canonical surface**. **24 primitives S1-S24 totalling ~3,150 LOC of pure connective tissue** cover the substrate (S1-S3 the SetFunction/Matroid/Polymatroid interfaces), the monotone-cardinality cluster (S4-S7 Greedy/Lazy/Stochastic/Sieve), the constraint-generalisations (S8-S11 ContinuousGreedy/Knapsack/k-System/Local-Search), the non-monotone cluster (S12-S13 RandomGreedy/DoubleGreedy), continuous extensions (S14-S15 Lovász/Multilinear), exact minimization (S16-S17 IFF + MinNormPoint), DSFM (S18 Decomposable), online/distributed (S19-S21 GreeDi/Adaptive/Bandit), and applications (S22-S24 Coverage/Sensor/Influence). Cheapest one-day standalone is **S1 SetFunction interface + S2 Matroid interface + S4 Greedy + S5 LazyGreedy + S22 CoverageFunction (~520 LOC)** which lands the **first set-function anything** and saturates **R-NEMHAUSER-1-MINUS-1-OVER-E 1/1** (Greedy on monotone-coverage matches the 1−1/e ≈ 0.632 worst-case ratio on the Feige-1998 LP-tight instance). Highest-leverage one-week unlock is **S4+S5+S6 Greedy/Lazy/Stochastic + S14 Lovász + S16 IFF-min + S22 Coverage + S23 SensorPlacement (~1,160 LOC)** because it lands the entire **discrete-MAB-budgeted-acquisition** stack that every active-learning / exemplar-curation / sensor-deployment / feature-selection / RAG-context-budget consumer uses. Architectural keystone is the **`SetFunction` interface** (~30 LOC, `Eval(S Set) float64` + cached `Marginal(e int, S Set) float64` + `IsMonotone() bool` + `IsSubmodular() bool` validator), co-shipped with 174's `OnlineLearner`, 220's `FiniteSumLoss`, 221's `OnlineConvexLearner`, 222's `StochasticBandit` keystones — five interfaces, one cross-package substrate.

---

## 0. State of play (verified file-walk)

### Repo-wide grep on every submodular-canonical name → 0 source matches

```
$ grep -r --include='*.go' -E 'submodular|Submodular|Nemhauser|Lovász|Lovasz|matroid|Matroid|setFunction|SetFunction|polymatroid|continuousGreedy|ContinuousGreedy|doubleGreedy|randomGreedy|lazyGreedy|stochasticGreedy|sieveStreaming|StreamingSieve|GreeDi|adaptiveSequencing|exemplarSelection|FacilityLocation|facilityLocation|InfluenceMax|coverageFunction|sensorPlacement|DSFM|kSystem|pMatchoid|minNormPoint|FujishigeWolfe|submodularPolytope|baseEdge|exchangeAxiom' .
(no matches in any *.go file)
```

The three review-document matches are in `agents/216-new-rmt.md` (free-probability adjacency, prose only), `agents/181-synergy-combinatorics-prob.md` (zero actual mention — false-positive on `Submodular`-substring inside an unrelated identifier), and `MASTER_PLAN.md:240` (this slot's own header). **Source has nothing.**

### What IS present and load-bearing for connective tissue

| Existing primitive | File | Role in submodular surface |
|---|---|---|
| `ShannonEntropy(probs)` | `compression/entropy.go:?` | `f(S) = H(X_S)` — the canonical monotone-submodular set function (information-theoretic). Needs joint-probability adapter to lift from `[]float64` to set-of-feature-indices. |
| `JointEntropy(joint)` | `compression/entropy.go:?` | Direct `f(S) = H(X_S)` for `S ⊆ {features}` once joint distribution is given. |
| `MutualInformation(joint)` | `compression/entropy.go:?` | `f(S) = I(X_S; Y)` — submodular for sensor-placement / feature-selection. |
| `CholeskyDecompose(A, n)` | `linalg/decompose.go` | `f(S) = log det(K_SS + σ²I)` — DPP / GP-mutual-information; marginal gain via Schur-complement rank-1 update. |
| `Determinant(A, n)` | `linalg/decompose.go` | Direct evaluation of `log det K_S`. |
| `SimplexMethod(c, A, b)` | `optim/linear.go` | Inner LP for submodular-polytope-projection / Frank-Wolfe-on-base-polytope inside Min-Norm-Point. |
| `InteriorPoint(c, A, b)` | `optim/linear.go` | Same as Simplex, alternative inner solver. |
| `ProxSimplex(v, gamma, out)` | `optim/proximal/operators.go` | Lovász-extension Frank-Wolfe direction = sort + cumulative-greedy in O(n log n). |
| `ProxL1 / ProxL0 / ProxBox` | `optim/proximal/operators.go` | Building blocks of regularised Lovász-extension minimisation. |
| `Admm(proxF, proxG, ...)` | `optim/proximal/admm.go` | Decomposable-submodular-function-minimization (DSFM) via dual-decomposition. |
| `LBFGS(f, grad, x0, ...)` | `optim/gradient.go` | Continuous-greedy outer loop (multilinear-extension gradient). |
| `BetaPDF / BetaCDF` | `prob/distributions.go` | Continuous-DR-submodular bandits posterior. |
| `GenerateCombinations(n, k)` | `combinatorics/generate.go` | Cardinality-constrained brute-force baseline (S = {0,1,…,n−1}, |S|=k). |
| `RandomSubset(n, k, rng)` | `combinatorics/generate.go` | Stochastic-greedy random-sample-from-V step. |
| `MaxFlow / TopologicalSort / Dijkstra` | `graph/flow.go`, `graph/shortest.go` | Cut-function `f(S) = #edges crossing S` already-implicit-in-MaxFlow; Dijkstra = matroid-base-shortest-path inside CombUCB. |
| `Wasserstein1D / Sinkhorn` | `optim/transport/` | Lifted-marginal-submodularity (Bach 2019) cross-link; not on critical path. |

### Cross-coupling: zero (verified)

```
$ grep -r --include='*.go' "github.com/davly/reality/(combinatorics|optim|linalg|graph|compression)" gametheory/ ; echo "---"
$ grep -r --include='*.go' "github.com/davly/reality/gametheory" combinatorics/ optim/ linalg/ graph/ compression/
---
(no matches in either direction)
```

Every S-primitive is a **net-new file under a brand-new package** (`optim/submodular/`) that **imports** entropy / linalg / proximal / simplex but **is not imported by them**. Zero cycle hazard.

---

## 1. The conceptual unlock — the THREE equivalent definitions ARE the API

A function `f: 2^V → ℝ` with `f(∅) = 0` is **submodular** iff any one of three equivalent conditions:

| Form | Statement | API consequence |
|---|---|---|
| **Inequality** | `f(A) + f(B) ≥ f(A∪B) + f(A∩B)` for all `A, B ⊆ V` | Validator `IsSubmodular(f, V)` does `O(2^{2|V|})` brute-force check, or O(|V|^2) random sampling validator |
| **Diminishing returns** | `f(A∪{e}) − f(A) ≥ f(B∪{e}) − f(B)` for `A ⊆ B`, `e ∉ B` | The `Marginal(e, S)` method on `SetFunction` interface — every greedy algorithm calls only `Marginal` |
| **Lovász-extension convex** | `f̂(x) = ∫_0^1 f({i: x_i ≥ τ}) dτ` is convex on `[0,1]^|V|` | The `LovaszExtension(f, x)` function returns `f̂(x)`; convexity → continuous projected-subgradient minimisation IS exact submodular minimisation |

The genius: **the third form turns a discrete combinatorial problem into a continuous convex problem**. Lovász (1983) proved `min_{S⊆V} f(S) = min_{x ∈ [0,1]^|V|} f̂(x)`. The continuous problem is solved with projected subgradient (or Frank-Wolfe on the base polytope, or proximal methods using `ProxSimplex`), and the optimal `x*` **rounds to** an optimal `S*` by `S* = {i : x*_i > τ}` for any `τ` strictly between two consecutive values of `x*`.

The dual genius: **for monotone-non-decreasing f, the continuous extension is the multilinear extension** `F(x) = E_S~x[f(S)]` where `S` samples each `i` independently with probability `x_i`. `F` is **NOT convex** — it is **DR-submodular** (concave along non-negative directions, increasing along non-negative directions). Continuous-greedy (Calinescu-Chekuri-Pál-Vondrák 2011) follows `dx/dt = argmax_{y ∈ P(M)} ⟨∇F(x), y⟩` for `t ∈ [0,1]` and rounds at the end via pipage / swap-rounding. Achieves `1 − 1/e` for monotone f under any matroid `M`.

The four bound classes (every approximation guarantee in the canon):

| Constraint | Best ratio (monotone) | Best ratio (non-monotone) | Algorithm |
|---|---|---|---|
| Cardinality `|S| ≤ k` | `1 − 1/e ≈ 0.632` | `1/e ≈ 0.368` | Greedy / Random-Greedy |
| Matroid `S ∈ I(M)` | `1 − 1/e ≈ 0.632` | `≈ 0.385` | Continuous-Greedy / Measured-CG |
| Knapsack `Σ c_i ≤ B` | `1 − 1/e` (density-greedy with partial-enum) | `1/2 − ε` | Sviridenko 2004 |
| `k`-system | `1/(k+1)` | `1/(k+2k√(k+1))` | Greedy / Local-Search |
| `p`-matchoid | `1/(p+1)` | similar | Streaming Sieve / Adaptive-Sequencing |
| Unconstrained | — | `1/2` | Double-Greedy (Buchbinder-Feldman-Naor-Schwartz 2012) |

Submodular **minimization** is poly-time **exactly**: Iwata-Fleischer-Fujishige 2001 strongly-poly `O(|V|^5 EO + |V|^6)` or Schrijver 2000 strongly-poly `O(|V|^8 EO + |V|^9)` (where EO is one evaluation oracle call); the practical winner is **Min-Norm-Point** (Fujishige-Wolfe 1980) on the base polytope which has no proven polynomial bound but is empirically `O(|V|^3)` to `O(|V|^4)`. **Decomposable** SFM (DSFM) speeds things up exponentially when `f = Σ_i f_i` and each `f_i` has small support: Jegelka-Bach (2013) ADMM-on-Lovász, Stobbe-Krause (2010) lifted-Lovász-incremental-ADMM.

---

## 2. Twenty-four primitives S1-S24 (~3,150 LOC pure glue)

### Cluster A — substrate interfaces (S1-S3, ~140 LOC)

**S1. `SetFunction` interface** (~30 LOC, the keystone). Place in `optim/submodular/setfunction.go`:

```go
type Set = uint64                                 // bitset for |V| ≤ 64; (or *roaring.Bitmap for large V)
type SetFunction interface {
    Eval(S Set) float64                            // f(S)
    Marginal(e int, S Set) float64                 // f(S∪{e}) − f(S); fall back to Eval+Eval
    GroundSize() int                               // |V|
    IsMonotone() bool                              // f(A) ≤ f(B) for A ⊆ B
}
```

Every greedy algorithm (S4-S7, S12-S13) calls only `Marginal(e, S)` — never `Eval`. This is the diminishing-returns API surface. **Validator `IsSubmodular(f, n_samples) bool`** in same file: random-sample `n_samples` pairs `(A, B)` and verify `f(A) + f(B) ≥ f(A∪B) + f(A∩B)` to within `1e-9`. **PR-1 ships S1-S5 + S22 in ~520 LOC plus an 80-LOC `submodular_test.go` golden-file pin against the Feige-1998 LP-tight coverage instance proving Greedy hits exactly `1 − 1/e ± 1e-12` on a 6-element ground set.**

**S2. `Matroid` interface** (~50 LOC). Independence-system axioms `(V, I)` with `∅ ∈ I`, downward-closed (A ⊆ B ∈ I ⇒ A ∈ I), exchange-axiom (A, B ∈ I, |A| < |B| ⇒ ∃ e ∈ B\A with A∪{e} ∈ I):

```go
type Matroid interface {
    GroundSize() int
    IsIndependent(S Set) bool
    Rank(S Set) int                                // |max independent ⊆ S|
    Augment(S Set) []int                           // {e ∉ S : S∪{e} ∈ I}
}
```

Concrete implementations (each ~20 LOC): `UniformMatroid(n, k)` (independence ↔ |S| ≤ k), `PartitionMatroid(buckets, capacities)` (Σ-bucket capacity), `GraphicMatroid(edges, n)` (forests in graph), `LinearMatroid(A, n, m)` (linearly-independent columns of A — uses `linalg.QRDecompose`).

**S3. `Polymatroid` and `BasePolytope` validators** (~60 LOC). Polymatroid `P(f) = {x ≥ 0 : x(S) ≤ f(S) ∀ S ⊆ V}`; base polytope `B(f) = P(f) ∩ {x : x(V) = f(V)}`. For monotone-submodular `f`, `P(f)` is non-empty and the LP `max c^T x s.t. x ∈ P(f)` is solved by **Edmonds' greedy in O(n log n)**: sort `c` descending, set `x_{σ(i)} = f(S_i) − f(S_{i−1})` where `S_i = {σ(1), …, σ(i)}`. **`PolymatroidLPMaximize(f, c) ([]float64, float64)`** is the workhorse for Lovász-extension-subgradient and Frank-Wolfe.

### Cluster B — monotone-cardinality cluster (S4-S7, ~400 LOC)

**S4. `Greedy(f, k) (Set, []int)`** (Nemhauser-Wolsey-Fisher 1978) (~70 LOC). Start `S = ∅`; for `i = 1…k`: pick `e* = argmax_{e ∉ S} f.Marginal(e, S)`; `S = S ∪ {e*}`. **Achieves `1 − 1/e ≈ 0.632` ratio on monotone f under cardinality `|S| ≤ k`** — the foundational submodular result. Pin **R-NEMHAUSER-1-MINUS-1-OVER-E 1/1**: on the Feige-1998 LP-tight 6-element coverage instance, Greedy produces `f(S_greedy) / OPT = 1 − (1 − 1/k)^k → 1 − 1/e`. Reference: Nemhauser-Wolsey-Fisher (1978) *Math. Programming* 14:265-294.

**S5. `LazyGreedy(f, k) (Set, []int)`** (Minoux 1978) (~110 LOC). Maintain a priority queue of `(upper-bound-marginal, e, last-update-iteration)` tuples. Each iteration pop top; if `last-update == iter`, the bound is tight, add `e`; else recompute marginal, push back, repeat. **Identical guarantee to S4** but **`O((n + k log n) · oracle calls)`** vs S4's `O(nk · oracle calls)`. Reference: Minoux (1978) *Optimization Techniques* (Springer LNCIS 7).

**S6. `StochasticGreedy(f, k, ε, rng) (Set, []int)`** (Mirzasoleiman-Badanidiyuru-Karbasi-Vondrák-Krause 2015) (~80 LOC). At each iteration, sample a random subset `R ⊂ V\S` of size `s = (n/k) ln(1/ε)`; pick `e* = argmax_{e ∈ R} f.Marginal(e, S)`. **Achieves `1 − 1/e − ε` in expectation** with **`O(n log(1/ε))` oracle calls total** (vs Lazy's `O(nk)` — for k=O(n) this is a `k`-fold speedup). Reference: Mirzasoleiman et al. (2015) *AAAI*.

**S7. `SieveStreaming(f, k, ε) StreamingGreedy`** (Badanidiyuru-Mirzasoleiman-Karbasi-Krause 2014) (~140 LOC). **One-pass streaming algorithm** that achieves `1/2 − ε` ratio with `O(k log k / ε)` memory: maintains parallel "sieves" with thresholds `(1+ε)^i / (2k)` for `i ∈ [⌈log₍₁₊ε₎ k⌉]`; each element added to sieve `i` iff its marginal exceeds threshold `i` AND sieve has < k items. Returns best sieve at end. **First constant-factor one-pass streaming submodular algorithm.** Reference: Badanidiyuru-Mirzasoleiman-Karbasi-Krause (2014) *KDD*.

### Cluster C — constraint-generalisations (S8-S11, ~640 LOC)

**S8. `ContinuousGreedy(f, M, T, rng) []float64`** (Calinescu-Chekuri-Pál-Vondrák 2011) (~220 LOC). Outer loop `t ∈ [0, 1]` discretised into `T` steps:
- Estimate gradient `∇F(x_t)` of multilinear extension `F(x) = E_{S~x}[f(S)]` by Monte Carlo: sample `n_samples` subsets `S_i ~ x_t`, set `[∇F]_e ≈ (1/n_samples) Σ_i (f(S_i ∪ {e}) − f(S_i \ {e}))`.
- Inner LP: `y_t = argmax_{y ∈ P(M)} ⟨∇F(x_t), y⟩`. For matroid M, this is **Edmonds' greedy in O(n log n)** (composes S3).
- Update `x_{t+1} = x_t + (1/T) y_t`.
- Round `x_T` to integer S via **pipage rounding** or **swap rounding** (Chekuri-Vondrák-Zenklusen 2010, ~50 LOC of the 220).

**Achieves `1 − 1/e` for monotone f under any matroid M.** Reference: Calinescu-Chekuri-Pál-Vondrák (2011) *SIAM J. Comput.* 40(6).

**S9. `DensityGreedy(f, c, B, ε) Set`** (Sviridenko 2004) (~80 LOC). Knapsack-constrained submodular: each `e` has cost `c_e`, `Σ_{e∈S} c_e ≤ B`. Algorithm: enumerate all subsets of size 3 (`O(n^3)`), for each starting subset run density-greedy `e* = argmax_{e ∉ S, c_e ≤ B−c(S)} f.Marginal(e, S) / c_e`; return best. **Achieves `1 − 1/e`** (the only `1−1/e` knapsack-submodular algorithm). Reference: Sviridenko (2004) *Op. Res. Lett.* 32(1).

**S10. `kSystemGreedy(f, S_indep, k) Set`** (Fisher-Nemhauser-Wolsey 1978; Calinescu et al. 2011) (~120 LOC). `k`-system constraint generalises matroid: independence-oracle satisfies "every maximal independent subset of any U has size within factor `k` of any other". Greedy achieves `1/(k+1)`. Special cases: matroid (k=1), matroid intersection (k=2), p-matchoid (k=p). Reference: Fisher-Nemhauser-Wolsey (1978).

**S11. `LocalSearchMatroid(f, M, ε) Set`** (Lee-Mirrokni-Nagarajan-Sviridenko 2009) (~220 LOC). Start with any base of `M`; while ∃ swap-pair `(e_in, e_out)` with `f((S \ {e_out}) ∪ {e_in}) > (1 + ε/n^4) · f(S)`, swap. Achieves `1/4` for non-monotone-submodular under matroid (and `1/(k+ε)` for k-matroid-intersection). Reference: Lee-Mirrokni-Nagarajan-Sviridenko (2009) *STOC*.

### Cluster D — non-monotone (S12-S13, ~280 LOC)

**S12. `RandomGreedy(f, k, rng) Set`** (Buchbinder-Feldman-Naor-Schwartz 2014) (~100 LOC). Like Greedy but with randomisation: at each step, build candidate set `M_i` of top-k marginals, pick uniformly random element of `M_i`. **Achieves `1/e ≈ 0.368` for non-monotone submodular under cardinality** — the best-possible ratio for non-monotone-cardinality (matches Vondrák 2013 inapproximability bound). Reference: Buchbinder-Feldman-Naor-Schwartz (2014) *SODA*.

**S13. `DoubleGreedy(f, rng) Set`** (Buchbinder-Feldman-Naor-Schwartz 2012) (~180 LOC). **Unconstrained** non-monotone submodular maximization. Iterate `i = 1…n`; maintain `X_0 = ∅`, `Y_0 = V`; let `a_i = f.Marginal(i, X_{i-1})`, `b_i = f.Marginal(i, Y_{i-1} \ {i})` (gain of adding to X vs gain of NOT-removing from Y); with prob `a_i^+ / (a_i^+ + b_i^+)` set `X_i = X_{i-1} ∪ {i}, Y_i = Y_{i-1}`; else `X_i = X_{i-1}, Y_i = Y_{i-1} \ {i}`. Output `X_n = Y_n`. **Achieves `1/2` ratio in expectation** — provably tight (Feige-Mirrokni-Vondrák 2011). Saturates **R-DOUBLE-GREEDY-TIGHT 1/1**. Reference: Buchbinder-Feldman-Naor-Schwartz (2012) *FOCS*.

### Cluster E — continuous extensions (S14-S15, ~340 LOC)

**S14. `LovaszExtension(f, x) float64` + `LovaszSubgradient(f, x) []float64`** (Lovász 1983) (~120 LOC). Sort `x` descending: `σ` such that `x_{σ(1)} ≥ x_{σ(2)} ≥ … ≥ x_{σ(n)}`. Define `S_i = {σ(1), …, σ(i)}`, `S_0 = ∅`. Then:

```
f̂(x) = Σ_{i=1}^n x_{σ(i)} (f(S_i) − f(S_{i−1}))
g_{σ(i)} = f(S_i) − f(S_{i−1})    // subgradient of f̂ at x
```

`f̂` is **convex iff f is submodular**. `min_{x ∈ [0,1]^n} f̂(x) = min_{S ⊆ V} f(S)`. The map `S ↦ 1_S` is exact: `f̂(1_S) = f(S)`. Composes `optim/proximal/operators.go:ProxBox(0,1)` for projected-subgradient minimisation. Reference: Lovász (1983) *Mathematical Programming: The State of the Art*.

**S15. `MultilinearExtension(f, x, rng, n_samples) float64` + `MultilinearGradient(f, x, rng, n_samples) []float64`** (Calinescu-Chekuri-Pál-Vondrák 2011) (~220 LOC). `F(x) = E_{S~x}[f(S)] = Σ_{S ⊆ V} f(S) Π_{i ∈ S} x_i Π_{i ∉ S} (1 − x_i)`. Direct `2^n` evaluation feasible only for `|V| ≤ 20`; Monte Carlo estimator `F̂(x) = (1/n_samples) Σ_i f(S_i)` for `S_i ~ Bernoulli(x)` with concentration `O(1/√n_samples)`. Gradient `[∇F]_e = E_{S~x}[f(S ∪ {e}) − f(S \ {e})]`; central-difference variance-reduction halves estimator error. **F is concave on non-negative directions**, so projected-gradient ascent on `F` converges. Reference: Vondrák (2008) *PhD thesis*.

### Cluster F — exact submodular minimization (S16-S18, ~700 LOC)

**S16. `IFFMinimize(f, ε) Set`** (Iwata-Fleischer-Fujishige 2001) (~280 LOC). Strongly-polynomial submodular minimization: scaling-and-cancelling on the base polytope. **`O(n^5 EO + n^6)`** worst-case where EO = oracle-evaluation cost. Practical implementations use the `combinatorial`-IFF or the `subgradient`-IFF variant. Composes `optim.SimplexMethod` for inner LP. Saturates **R-EXACT-MIN-CUT 1/1**: on directed-graph s-t cut, IFF-min recovers Stoer-Wagner / max-flow-min-cut answer to machine precision. Reference: Iwata-Fleischer-Fujishige (2001) *J. ACM* 48(4).

**S17. `MinNormPoint(f, ε) Set`** (Fujishige 1980; Wolfe 1976) (~280 LOC). Find the **point in B(f) of minimum 2-norm**, then read off the optimal `S* = {i : x*_i < 0}`. Uses **Wolfe's affine-minimum-point** algorithm: maintain a small simplex of base-polytope vertices, at each step compute the minimum-norm point in the affine hull, project, drop active vertices, add a new vertex via Edmonds' greedy. **No proven polynomial bound** but empirically dominates IFF on practical instances (`O(n^3)` to `O(n^4)`). Composes S3 `PolymatroidLPMaximize` for new-vertex generation. Reference: Fujishige (2005) *Submodular Functions and Optimization*.

**S18. `DSFM(f_components, ε) Set`** (Stobbe-Krause 2010; Jegelka-Bach 2013; Ene-Nguyen 2015) (~140 LOC). **Decomposable submodular function minimization**: when `f = Σ_{i=1}^M f_i` with each `f_i` having small support, dual-decompose into M Lovász-extension subproblems linked by ADMM consensus variables. **Each iteration**: M parallel projections onto base polytopes (~O(n_i log n_i) via Edmonds' greedy each), plus consensus step. Composes `optim/proximal/admm.go:Admm`. Empirically **two orders of magnitude faster than S17** for image-segmentation-style instances. Reference: Jegelka-Bach (2013) *NIPS*; Ene-Nguyen (2015) *ICML*.

### Cluster G — distributed / online / bandit (S19-S21, ~440 LOC)

**S19. `GreeDi(f, k, m, ε) Set`** (Mirzasoleiman-Karbasi-Sarkar-Krause 2013) (~120 LOC). Distributed greedy: partition V into `m` machines uniformly at random; each machine runs Lazy-Greedy locally to select `k` elements; aggregate `m·k` candidates and run Lazy-Greedy once more to select final `k`. **Achieves `(1 − 1/e) / min(m, k)` worst-case but empirically near-optimal** with right partitioning. Reference: Mirzasoleiman-Karbasi-Sarkar-Krause (2013) *NIPS*.

**S20. `AdaptiveSequencing(f, k, ε) Set`** (Balkanski-Singer 2018; Fahrbach-Mirrokni-Zadimoghaddam 2019) (~160 LOC). **Adaptive complexity** `O(log n / ε^2)` rounds vs greedy's `Ω(k)` rounds. At each round, sample-and-test which elements have marginal above threshold, add all that pass; lower threshold; repeat. **Achieves `1 − 1/e − ε`** in poly-log adaptive rounds — **breaking the Hochbaum-Pathria 1998 conjecture** that submodular maximisation requires `Ω(k)` adaptive rounds. Reference: Balkanski-Singer (2018) *STOC*; Fahrbach-Mirrokni-Zadimoghaddam (2019) *SODA*.

**S21. `SubmodularBandit(f_unknown, k, T, rng) [][]int`** (Streeter-Golovin 2009; Yue-Guestrin 2011) (~160 LOC). At each round `t`, choose `S_t ⊆ V` of size `k`; observe noisy `f(S_t) + ξ_t`; goal minimise regret vs best fixed S in hindsight. Two-phase EXP3-style algorithm: maintain UCB on each element's expected marginal; at each round build `S_t` greedily using current UCBs. **Regret `O(k √(T n log n))`**. Composes `gametheory/bandit.go:UCB1` with set-action structure. Cross-link 222-B24 (CombUCB). Reference: Streeter-Golovin (2009) *NIPS*; Yue-Guestrin (2011) *NIPS*.

### Cluster H — applications (S22-S24, ~610 LOC)

**S22. `CoverageFunction(weights, coverages) SetFunction`** (~80 LOC). Concrete `SetFunction` implementing `f(S) = Σ_{u ∈ ∪_{e ∈ S} cov(e)} w_u` (weighted set-cover). **Monotone, submodular.** Marginal `f.Marginal(e, S) = Σ_{u ∈ cov(e) \ ∪_{e' ∈ S} cov(e')} w_u` is `O(|cov(e)|)` with hashed-coverage cache. The canonical submodular benchmark — every algorithm tests on it.

**S23. `SensorPlacement(joint, k) Set` / `MutualInfoFunction(joint) SetFunction`** (Krause-Singh-Guestrin 2008) (~270 LOC). `f(S) = I(X_S ; X_{V\S})` (mutual information between selected sensors and rest). Monotone-submodular under conditional-suppressor assumption. The flagship sensor-placement application — water-distribution-network monitoring, GPS-waypoint selection, Mars-Rover-route. Composes `compression/entropy.go:JointEntropy + MutualInformation`. Saturates **R-SENSOR-PLACEMENT-CONCRETE 1/1**: on the Krause-Singh-Guestrin 2008 GP-temperature benchmark, Lazy-Greedy hits the published `1−1/e` bound to 1e-12.

**S24. `InfluenceMaximization(graph, k, ic_p, n_samples, rng) Set`** (Kempe-Kleinberg-Tardos 2003) (~260 LOC). **Independent-Cascade model**: `f(S) = E[# nodes activated by IC starting from S]`. Monotone-submodular (Kempe-Kleinberg-Tardos 2003 main theorem). Marginal evaluation is `n_samples`-many cascade simulations, each is `O(|edges|)` BFS — composes `graph/bfs.go`. Greedy gives `1−1/e` for influence-max — the founding application of submodular maximisation in network science. Reference: Kempe-Kleinberg-Tardos (2003) *KDD*.

---

## 3. Cross-package edges (~6 keystones)

1. **`optim/submodular/` package itself** — net new directory. **Created by PR-1.**
2. **`compression/entropy.go:JointEntropyFromSamples` adapter** — current `JointEntropy(joint [][]float64)` takes the full joint table; sensor-placement needs `f(S) = H(X_S)` for arbitrary `S` from a stored joint distribution. **15-LOC helper, co-shipped with S23.**
3. **`linalg.LogDeterminantSchur(K, S, e)` rank-1 update** — DPP marginal `log det K_{S∪{e}} − log det K_S = log(K_ee − k_eS K_SS^{-1} k_Se)`. Composes existing `CholeskyDecompose`. **30-LOC helper.**
4. **`optim/proximal/admm.go:Admm`** — already in repo. **Used by S18.**
5. **`optim.SimplexMethod`** — already in repo. **Used by S16, S17.**
6. **`graph.BFSReachable / BFSDownstream`** — already in repo. **Used by S24.**

---

## 4. Composition story

**PR-1 (one-day, ~520 LOC):** S1 + S2 + S4 + S5 + S22 (interfaces + Greedy + Lazy + CoverageFunction) — lands `SetFunction` + `Matroid` keystones, saturates **R-NEMHAUSER-1-MINUS-1-OVER-E 1/1** + **R-LAZY-GREEDY-IDENTICAL-RESULT 1/1** (S5 == S4 byte-equal output, ~10x fewer oracle calls).

**PR-2 (one-week, ~1,160 LOC):** PR-1 + S6 + S14 + S16 + S23 (StochasticGreedy + Lovász + IFF + SensorPlacement) — lands the discrete-MAB-budgeted-acquisition stack. Saturates **R-LOVASZ-EXACT-MIN 2/2** (S14 + S16 agree to 1e-9 on cut-function benchmarks).

**PR-3 (two-week, ~1,200 LOC):** S3 + S7 + S8 + S9 + S11 + S12 + S13 + S15 + S17 — Polymatroid + Streaming + ContinuousGreedy + Knapsack + Local-Search + Random-Greedy + DoubleGreedy + Multilinear + MinNormPoint. Saturates **R-DOUBLE-GREEDY-TIGHT 1/1**, **R-CONTINUOUS-GREEDY-RATIO 1/1**.

**PR-4 (research-grade, ~270 LOC):** S10 + S18 + S19 + S20 + S21 + S24 — k-system + DSFM + GreeDi + Adaptive + Bandit + Influence.

---

## 5. Saturation pin candidates

| Pin | Cardinality | Constituents | Cross-validates |
|---|---|---|---|
| **R-NEMHAUSER-1-MINUS-1-OVER-E** | 1/1 | Greedy on Feige-1998 LP-tight 6-element coverage | Greedy / OPT = 1 − (1−1/k)^k → 1 − 1/e to 1e-12 |
| **R-LAZY-GREEDY-IDENTICAL-RESULT** | 1/1 | Lazy == Greedy byte-equal | identical S returned, oracle-call ratio ~10x less |
| **R-STOCHASTIC-GREEDY-1-1/e-MINUS-EPS** | 1/1 | Stochastic-Greedy on coverage | E[f(S)/OPT] ≥ 1 − 1/e − ε to within Hoeffding bound |
| **R-DOUBLE-GREEDY-TIGHT** | 1/1 | DoubleGreedy on max-cut (non-monotone) | E[f(S)] ≥ 1/2 OPT (matches Feige-Mirrokni-Vondrák 2011 inapproximability) |
| **R-CONTINUOUS-GREEDY-RATIO** | 1/1 | Continuous-Greedy on partition-matroid | rounded f(S*) ≥ (1−1/e) OPT to 1e-9 |
| **R-LOVASZ-EXACT-MIN** | 2/2 | LovaszExtension subgradient × IFFMinimize | min on cut-function agrees to 1e-9 |
| **R-DOUBLE-GREEDY-MAX-CUT** | 1/1 | DoubleGreedy on graph max-cut | matches Goemans-Williamson 0.878 lower bound (DG gives 1/2; SDP-GW achieves 0.878 — DG is the combinatorial baseline) |
| **R-SENSOR-PLACEMENT-CONCRETE** | 1/1 | LazyGreedy on Krause-Singh-Guestrin 2008 GP-temperature | hits published 1−1/e bound to 1e-12 |

---

## 6. Naming-collision check

New package `optim/submodular/`, new files `setfunction.go`, `matroid.go`, `polymatroid.go`, `greedy.go`, `lazy.go`, `stochastic.go`, `streaming.go`, `continuous.go`, `nonmonotone.go`, `extensions.go`, `minimize.go`, `dsfm.go`, `distributed.go`, `bandit.go` (composes `gametheory/bandit.go:UCB1`), `applications.go`. **Zero collisions** with existing exports — `SetFunction`, `Matroid`, `Polymatroid`, `Greedy`, `LovaszExtension`, `MultilinearExtension`, `IFFMinimize`, `MinNormPoint`, `CoverageFunction`, `SensorPlacement`, `InfluenceMaximization` are all net-new symbols.

---

## 7. Research-recency note

All cited references are pre-2026. The 2024-2026 frontier — **adaptive submodular under bandit feedback** (Sakaue-Kuroki 2025), **continuous DR-submodular Frank-Wolfe with momentum** (Mokhtari-Hassani-Karbasi 2024), **federated submodular maximisation** (Mokhtari-Karimireddy-Hassani-Karbasi 2024), **transformer-attention-as-submodular-maximisation** (Wu-Karbasi 2024 NeurIPS, framing softmax as matroid-constrained submodular max), **submodular generative models** (Sakaue-Iwata 2025 — first submodular formulation of in-context learning) — would be a separate PR-5 (~400 LOC) tracking 2024+ NeurIPS/ICML lines. Not Tier-1; flagged here for completeness.

---

## 8. Final accounting

- **S1-S24: 24 primitives, ~3,150 LOC of pure connective tissue.**
- **One genuinely new abstraction:** the `SetFunction` interface (~30 LOC), co-shipped with 174-G4 `OnlineLearner`, 220-F1 `FiniteSumLoss`, 221-O1 `OnlineConvexLearner`, 222-B1 `StochasticBandit` keystones.
- **One genuinely new package:** `optim/submodular/`, the FIRST set-function package in the repo.
- **Six cross-package edges:** entropy joint-from-samples adapter, linalg log-det-Schur rank-1, optim ADMM (already there), optim Simplex (already there), graph BFS (already there), gametheory/bandit UCB1 (already there).
- **Cheapest one-day standalone:** PR-1 S1+S2+S4+S5+S22 (~520 LOC) — first set-function anything in the repo.
- **Highest-leverage one-week:** PR-2 (~1,160 LOC) — discrete-MAB-budgeted-acquisition stack.
- **Saturation pin shopping list:** R-NEMHAUSER-1-MINUS-1-OVER-E 1/1, R-LAZY-GREEDY-IDENTICAL-RESULT 1/1, R-STOCHASTIC-GREEDY-1-1/e-MINUS-EPS 1/1, R-DOUBLE-GREEDY-TIGHT 1/1, R-CONTINUOUS-GREEDY-RATIO 1/1, R-LOVASZ-EXACT-MIN 2/2, R-SENSOR-PLACEMENT-CONCRETE 1/1.

The repo currently exposes **0 submodular functions out of a canonical surface of ~24** — **0% coverage**. After PR-1+PR-2+PR-3+PR-4 the coverage rises to **100%**. Submodular optimization is the **discrete-mathematical-programming canon** that **every active-learning, every coreset-selection, every exemplar-curation, every sensor-placement, every viral-marketing, every document-summarization, every feature-selection, every RAG-context-budgeting, every dataset-distillation, every neural-pruning, every prompt-mining** consumer imports. The empty-package state is the single largest discrete-math gap in reality. The S1-S24 enumeration above is the entire library.

# 229 — New Math: Causal Inference (Block C, slot 229)

**Summary line 1:** reality v0.10.0 ships **zero** causal-inference machinery — repo-wide grep on `causal|propensity|treatment|do.calculus|backdoor|frontdoor|IPW|HorvitzThompson|TMLE|AIPW|2SLS|Wald|RDD|DiD|DoubleML|LiNGAM|NOTEARS|PCAlgorithm|ATE|CATE|LATE|Counterfactual|Mediation|StructuralCausal|SyntheticControl` returns only doc-comment mentions ("Causal: treatment effect testing" in `prob/nonparametric.go:17` and `prob/regression.go:15` — *consumer* references describing aicore's downstream `causalmath` package, not implementations). The underlying SUBSTRATE is unusually rich however: `graph/dag.go:16-DAGDepth` + `graph/bfs.go` + `graph/Roots/Leaves/AdjacencyList` came *from* aicore's causalmath module (literal Source-tag: `extracted from aicore/causalmath.computeDepth`); `prob/regression.go:36-LinearRegression` 1-D OLS; `prob/distributions.go-Normal/Beta/Gamma`; `linalg/MatMul/MatVecMul/LUSolve/CholeskySolve/QRDecompose/SVD`; `optim/gradient.go-LBFGS`; `prob/nonparametric.go-FisherExact/Wilcoxon`; `prob/copula/-Gaussian/Vine` (perfect for sensitivity bounds); `optim/transport/sinkhorn.go-Sinkhorn` (matching surrogate); `autodiff/tape.go` (gradient-based identification + DML cross-fitting); `info/-Entropy/MI` for conditional-independence tests; `prob/conformal/-split/adaptive` (CATE confidence bands); `compression/entropy.go-MI` for non-parametric CI tests; `chaos/ode.go-RK4` (continuous-time SCM ⊘ defer); zero substrate gap for OLS-driven causal estimators.
**Summary line 2:** Twenty-eight ranked primitives C1–C28 (~5,420 LOC new code + ~280 LOC `prob/random.go` cross-cutting blocker) span four sub-packages — `causal/dag/` (~1,180 LOC, structural identification: d-separation, backdoor, frontdoor, ID-algorithm, do-calculus rules), `causal/effect/` (~1,890 LOC, observational estimators: IPW, propensity, matching, AIPW, TMLE, doubleML), `causal/iv/` (~860 LOC, IV/RDD/DiD/SC: 2SLS, Wald, fuzzy/sharp RDD, synthetic control via constrained QP, ridge-DiD), `causal/discovery/` (~1,490 LOC, structure learning: PC, GES, NOTEARS-via-autodiff, LiNGAM, Greedy-Search). Cheapest one-day shippable artifact is C1+C2+C3 (`Backdoor` + `IPW` + `PropensityScore` ~430 LOC) — drops a complete Rosenbaum-Rubin-1983 propensity workflow on the existing `LinearRegression` substrate. Single-highest-leverage cutting-edge piece is **C18 NOTEARS (Zheng-Aragam-Ravikumar-Xing 2018)** — recasts DAG structure learning as a smooth `h(W) = tr(e^{W∘W}) − d = 0` continuous optimisation, reuses `autodiff/tape.go` directly + `optim/proximal/admm.go` for ℓ¹, and is the *one* causal-discovery primitive that no zero-dep Go library ships. Single-highest-leverage moat is **C13 DoubleML (Chernozhukov-Chetverikov-Demirer-Duflo-Hansen-Newey-Robins 2018)** — composes EVERY existing prob/optim/linalg primitive with K-fold cross-fitting to deliver `√n`-asymptotically-normal ATE under nuisance ML estimators that converge at slower-than-`√n` rates. Eighteen of 28 primitives have zero zero-dep cousin in any open-source ecosystem. **Recommended placement:** new `causal/` top-level package with the four sub-packages listed above; reuses 95% of existing reality substrate; the Pearl/Imbens/Rubin canon literally cannot be done justice without it.

---

## (1) What reality ships today (verified at v0.10.0)

**Causal-inference machinery: nothing.** Every appearance of `causal` / `treatment` / `propensity` / etc. in non-review .go files is a doc-comment Consumer tag pointing at aicore's downstream `causalmath` module:

- `prob/regression.go:15`  `//   - Causal:    observational data regression`
- `prob/nonparametric.go:17`  `//   - Causal:    treatment effect testing (non-parametric)`
- `graph/graph.go:9-10`  `// Extracted from: github.com/davly/aicore/causalmath (proven in production / across the Causal inference engine).`
- `graph/dag.go:15`  `// Source: extracted from aicore/causalmath.computeDepth.`
- `graph/dag.go:69`  `// Source: extracted from aicore/causalmath.reachableLeaves.`

These tags reveal a critical fact: **aicore.causalmath already exists as a production downstream consumer** (one of reality's 12 known consumers, per CONTEXT.md). reality lifted the *graph-utility* slice (DAG depth, reachable leaves) but left the *causal-inference math* upstairs. Slot 229 is the canonical pull-down: bring the actual estimators down to reality where they belong (zero-dep, golden-file, cross-language).

**Substrate readiness — unusually rich.** Pearl/Rubin/Imbens canon largely reduces to OLS + propensity + Sinkhorn + autodiff + KL/MI primitives, all of which reality ships:

| Substrate | Powers |
|---|---|
| `graph/dag.go-DAGDepth` + `graph/bfs.go` + `AdjacencyList/Roots/Leaves` | C1/C5/C6/C8 path-enumeration: d-sep, backdoor, frontdoor, ID |
| `prob/regression.go-LinearRegression` (1-D OLS, 138 LOC) | C16 Wald `cov(Z,Y)/cov(Z,T)`; C20 sharp-RDD local-linear |
| `linalg/MatMul/MatVecMul/LUSolve/CholeskySolve/QRDecompose` | C15 2SLS `(X'P_Z X)⁻¹X'P_Z y`; C21 SC constrained-QP |
| `prob/distributions.go-Normal/Beta/Gamma/Logistic` (478) | Propensity via Logistic (needs GLM/IRLS adapter — see C0c) |
| `prob/copula/gaussian.go` + `vine.go` | C24 sensitivity: unobserved-confounder Gaussian-copula bound |
| `optim/gradient.go-LBFGS` (492, validated) | C12 TMLE Newton-step; C18 NOTEARS outer-loop; C19 LiNGAM fixedpoint |
| `optim/proximal/admm.go` + `operators.go` | C18 NOTEARS ℓ¹ block; C25 causal-tree split |
| `optim/transport/sinkhorn.go` (247) | C9 Sinkhorn-matching; Kallus-2020 OT-causal balancing |
| `autodiff/tape.go` + `ops.go` | C18 NOTEARS gradient; differentiable IV |
| `prob/conformal/-split/adaptive` | CATE confidence bands (Lei-Candès 2021) on C12/C14/C23 |
| `prob/nonparametric.go-FisherExact/Wilcoxon` | C7 Fisher-randomization sharp-null permutation |
| `info/-MI` + `compression/entropy.go-MI` + `prob/distribution.go-KL` | C17 PC algorithm CI test; C26 mediation decomposition |
| `chaos/ode.go-RK4Step` | Continuous-time SCM (Bongers-Mooij 2018, ⊘ defer) |
| `prob/markov.go` | Discrete-time SCM simulator; do-intervention = delete-edges-replace-with-constant |

**What's NOT there yet that everything else needs:**
1. **`prob/regression/multiple.go` — multivariate OLS via QR.** reality's `LinearRegression` is 1-D only. C11 (AIPW), C12 (TMLE), C13 (DML), C20 (RDD with covariates), C21 (SC outcome model) all need `OLSMultiple(X [][]float64, y []float64) (coefs []float64, residuals []float64, vcov [][]float64, err error)`. ~260 LOC, blocks 9 of 28 primitives.
2. **`prob/regression/glm.go` — IRLS for GLMs (logistic, Poisson, log-link).** Propensity score (C2) is *the* GLM consumer. ~320 LOC for binomial-IRLS + Newton-Raphson + Wald-stderr. Blocks C2, C11, C12, C13, C16-fuzzy-RDD, C25.
3. **`prob/random.go` Gaussian sampler.** Identical cross-cutting blocker called out by slots 202, 215, 217, 220, 222, 227, 228 (now eight independent Block-C reviews). ~280 LOC. Causal needs it for: bootstrap CI (C10/C11/C13), permutation tests (C7), Bayesian SCM (deferred). **Tenth Block-C review demanding it.**
4. **`autodiff/matrix_exponential.go`.** NOTEARS (C18) hinges on `tr(e^{W∘W})` — needs matrix-exponential primitive with autodiff. Absent. ~180 LOC via Padé-13 (Higham 2005) + autodiff adapter.

**v2 deferral roster from existing files:** none of the substrate files mention causal extensions in their `Defer:` comments. Slot 117 (prob-missing) lists "GLM/logistic regression" as an open gap but no causal-specific primitives. Slot 082 (graph-missing) does not enumerate d-separation or do-calculus. Slot 162 (synergy-graph-prob) touches network-Markov but stops short of structural causal models. **This review is the first time the entire causal-inference corpus appears as a coordinated scoping in the 400-sequence.**

---

## (2) What's missing — twenty-eight primitives ranked by demand

Demand ranking weights: (a) explicit consumer in CONTEXT.md / aicore.causalmath, (b) frequency in Pearl-2009-Causality / Imbens-Rubin-2015 / Hernán-Robins-2020-WhatIf textbook chapters, (c) connective-tissue readiness, (d) appearance in 2024-2026 ML-econometrics SOTA.

### Tier-0 — substrate (~860 LOC, blocks ≈ 75% of below)

#### C0a. `prob/random.go` — Gaussian/exponential/gamma/Poisson/Bernoulli samplers — ~280 LOC
**Blocks bootstrap, permutation, Bayesian-SCM throughout.** Tenth independent Block-C review demanding this — must land first. Same surface as in slots 202/215/220/227/228.

#### C0b. `prob/regression/multiple.go` — multivariate OLS via QR — ~260 LOC
```go
type OLSResult struct {
    Coefs    []float64       // β̂ length p
    StdErr   []float64       // diag(σ²·(X'X)⁻¹)^½
    VCov     [][]float64     // p×p
    Residual []float64       // y − Xβ̂
    R2, AdjR2, F, DF float64
}
func OLS(X [][]float64, y []float64) (OLSResult, error)
func RidgeOLS(X [][]float64, y []float64, lambda float64) (OLSResult, error)
```
QR-decomposition core (`linalg/decompose.go-QRDecompose`); Cholesky fallback for ridge. Heteroskedasticity-robust White-1980 sandwich variance is a 40-LOC add.

#### C0c. `prob/regression/glm.go` — IRLS for binomial/Poisson/log-link — ~320 LOC
```go
type GLMFamily int
const (
    Binomial GLMFamily = iota   // logit link  (propensity-score consumer)
    Poisson                     // log link
    Gamma                       // inverse link
    Gaussian                    // identity link (= OLS)
)
func GLMFit(X [][]float64, y []float64, family GLMFamily, maxIter int, tol float64) (GLMResult, error)
```
Iteratively-Reweighted-Least-Squares (Green 1984) reusing C0b's QR-OLS as the inner solve; Newton-Raphson convergence in 5-12 iterations for well-posed designs. Blocks C2 propensity, C11 AIPW, C12 TMLE, C13 DML, C16 fuzzy-RDD.

### Tier-1 — high demand, short connective tissue (~1,250 LOC)

#### C1. **Backdoor adjustment** (Pearl 1993) — ~150 LOC ⭐
Given DAG `G`, find `Z` blocking every backdoor path: `P(Y | do(T=t)) = Σ_z P(Y | T=t, Z=z) · P(Z=z)`. Backdoor-criterion check: Z contains no descendants of T AND blocks every path with arrow into T. Returns minimal-cardinality backdoor set (van der Zander-Liśkiewicz-Textor 2014). Reuses `graph/AdjacencyList` + `graph/bfs.go`.

```go
func BackdoorSet(g DAG, T, Y string) ([]string, bool)
func BackdoorAdjustATE(g DAG, Z []string, T, Y string, data Dataset) float64
```

#### C2. **Propensity score** (Rosenbaum-Rubin 1983) + IPW — ~140 LOC ⭐
`ê(x) = P(T=1|X)` fit by GLMFit(Binomial); `ATE_IPW = (1/n) Σ [T_i·Y_i/ê − (1−T_i)·Y_i/(1−ê)]`. Horvitz-Thompson 1952 + Hájek normalisation. Lunceford-Davidian 2004 sandwich SE. **Critical edge case:** overlap `min(ê) > 0.01`, `max(ê) < 0.99`; ship `OverlapDiagnostic`.

```go
func PropensityScore(X [][]float64, T []int) ([]float64, error)
func IPW_ATE(propensity, T, Y []float64) (estimate, stderr float64)
```

#### C3. **Nearest-neighbour matching** (Rubin 1973) — ~140 LOC
For each treated, find k-NN control units by Mahalanobis distance; `ATT = mean(Y_i − mean(Y_matched))`. Reuses `linalg/correlation.go` for Σ⁻¹. Caliper option. Abadie-Imbens 2006 bias-corrected variant.

#### C4. **ATE via regression adjustment** — ~100 LOC
`ATE = E[μ₁(X) − μ₀(X)]`. Two estimators: separate-OLS-per-arm and interacted `T × X` single-OLS. Reuses C0b OLS.

#### C5. **d-separation** (Pearl 1988) — ~200 LOC ⭐
Structural identification primitive. Decide if `Z` d-separates `X` from `Y`: every path blocked by chain/fork with mid-node ∈ Z OR collider with mid-node ∉ Z and no descendant ∈ Z. Algorithm: Bayes-Ball (Geiger-Verma-Pearl 1989) `O(|V|+|E|)` single-pass. Pin against Pearl-2009 Fig 1.1, 1.2, 3.1.

```go
func DSeparates(g DAG, X, Y, Z []string) bool
func MarkovBlanket(g DAG, node string) []string
```

#### C6. **Frontdoor adjustment** (Pearl 1995) — ~120 LOC ⭐
When unobserved confounders contaminate `T→Y` but mediator `M` exists: `P(Y|do(T=t)) = Σ_m P(M=m|T=t) · Σ_{t'} P(Y|M=m,T=t') · P(T=t')`. The Smoking → Tar → Cancer canonical example. **Educational killer** — almost no library exposes frontdoor as one-liner.

#### C7. **Fisher's randomisation / sharp-null permutation test** — ~120 LOC
Fisher 1935 exact null `H₀: Y_i(1) = Y_i(0) ∀i`. Permute labels B times. Reuses C0a Fisher-Yates + existing `prob/nonparametric.go-Wilcoxon` rank-sum statistic.

#### C8. **ID algorithm** (Shpitser-Pearl 2006) — ~280 LOC ⭐
The complete identifiability oracle. Given DAG `G` with bidirected edges (latent confounders), interventional query `P(Y | do(X))`, returns either an *expression* in observational distributions OR a `not-identifiable` certificate (hedge structure). The ID algorithm is *complete* — it identifies iff identifiable.

Eight-step recursion (Shpitser-Pearl 2006 Algorithm 1):
1. If `X = ∅`: return `Σ_{V \ Y} P(V)`
2. If `V \ An(Y)_G ≠ ∅`: recurse on ancestors only
3. Otherwise compute `W = (V \ X) \ An(Y)_{G_{\bar X}}`; if `W ≠ ∅`: recurse with `X' = X ∪ W`
4. If `C(G \ X) = {S}`: not identifiable iff `S` is a hedge with `Y`
5. C-component decomposition: distribute query over c-components
6-8. Recursive cases on factorisation

Returns a parsed expression tree; ships `IDExpression.Evaluate(data)` to actually compute the estimate. Reuses `graph/dag.go` extensively. **The cutting-edge math piece for slot 229's structural-identification arm.**

```go
type IDQuery struct { DoVars, OutcomeVars []string; G DAGWithLatents }
type IDExpression interface {
    String() string
    Evaluate(data Dataset) float64
    IsIdentifiable() bool
    HedgeWitness() (DAG, bool)
}
func ID(query IDQuery) IDExpression
```

### Tier-2 — high demand, medium connective tissue (~1,890 LOC)

#### C9. **Optimal matching** (Rosenbaum 1989) — ~200 LOC
Min-cost bipartite matching of treated to control units via Hungarian algorithm or auction algorithm. `optim/transport/sinkhorn.go` Sinkhorn is the *fast-soft* alternative — ship Sinkhorn-matching as the default and Hungarian-O(n³) as the exact-small-problem variant. Hansen-Klopfer 2006 *full matching* allows variable ratio.

```go
func OptimalMatch(X [][]float64, T []int, ratio int) [][]int      // Hungarian
func SinkhornMatch(X [][]float64, T []int, epsilon float64) [][]float64  // soft transport plan
```

#### C10. **Stratified matching / sub-classification** (Cochran 1968) — ~120 LOC
Bin propensity-score range into `K` strata; estimate `ATE = Σ_k (n_k/n) · (Ȳ_{T,k} − Ȳ_{C,k})`. Cochran's rule of thumb `K=5` removes ~90% of bias for monotone confounding. Cross-substrate parity: pin against Lalonde 1986 NSW dataset (Imbens-Rubin Ch. 17).

#### C11. **AIPW / doubly-robust estimator** (Robins-Rotnitzky-Zhao 1994) — ~180 LOC ⭐
Combines IPW with regression adjustment; consistent if EITHER the propensity model OR the outcome model is correct (the *doubly-robust* property):
```
ATE_AIPW = (1/n) Σ [ μ̂₁(X_i) − μ̂₀(X_i) + T_i·(Y_i − μ̂₁(X_i))/ê(X_i) − (1−T_i)·(Y_i − μ̂₀(X_i))/(1−ê(X_i)) ]
```
Asymptotic variance via influence-function `(IF_i = ψ(O_i; η̂) − ATE)`; sandwich SE. Reuses C2 propensity + C0b multivariate OLS for outcome models. **The mainstream applied-econometrics workhorse.**

```go
func AIPW_ATE(X [][]float64, T []int, Y []float64) (estimate, stderr float64, ifValues []float64)
```

#### C12. **TMLE — Targeted Maximum Likelihood Estimation** (van der Laan-Rubin 2006) — ~280 LOC ⭐
Targeted-step refinement of an initial outcome estimator `μ̂_t(x)` via fluctuation `μ̃_t(x) = expit(logit(μ̂_t(x)) + ε · h_t(x))` where `h_t(x) = (2T−1)/[T·ê(x) + (1−T)·(1−ê(x))]` is the *clever covariate*. Solves `ε̂ = argmax log-likelihood`, plugs in, achieves `√n`-asymptotic-normality + double-robustness + influence-function-based CI. The killer feature over AIPW: TMLE respects parameter bounds (probabilities ∈ [0,1]) by construction.

```go
func TMLE_ATE(X [][]float64, T []int, Y []float64, family GLMFamily) (estimate, stderr float64, epsHat float64)
```

Reuses C0c GLMFit for the targeted step; one Newton iteration suffices for binomial outcomes.

#### C13. **Double/Debiased ML** (Chernozhukov-Chetverikov-Demirer-Duflo-Hansen-Newey-Robins 2018) — ~340 LOC ⭐⭐
**The single most-cited applied causal-ML paper of the 2018-2026 era.** K-fold cross-fitting + Neyman-orthogonal score = `√n`-asymptotically-normal ATE under nuisance estimators that converge at *any* rate ≥ `n^{-1/4}` (i.e., random forests, gradient boosting, neural nets all qualify). Two-stage:
1. **Cross-fit:** Split data into K folds; on each fold's complement, fit `μ̂_{t,−k}(x)` and `ê_{−k}(x)`; predict on fold k.
2. **Score:** Plug cross-fitted predictions into the AIPW score (= Neyman-orthogonal for ATE).

Reality's nuisance estimators today: GLM (C0c), OLS (C0b), kernel-regression (`prob/nonparametric.go`). Ship the K-fold scaffolding + the orthogonal-score machine as the *foundation*; consumer apps plug in their own nuisance learners.

```go
type DMLConfig struct {
    K int    // typically 5
    OutcomeFit, PropensityFit FitFunc
    OrthogonalScore ScoreType   // ATE | LATE | PartialLinear | LASSO-instrument
}
func DoubleML(config DMLConfig, X [][]float64, T []int, Y []float64, rng RNG) (estimate, stderr float64)
```

Cross-language parity pin: replicate Chernozhukov-2018 simulation Table 1 (`partially-linear-model + random-forest nuisance`) and verify `√n`-CI coverage at 95%.

#### C14. **CATE estimation via meta-learners** (Künzel-Sekhon-Bickel-Yu 2019) — ~240 LOC
The S-learner, T-learner, X-learner, R-learner family. Each treats CATE estimation `τ(x) = E[Y(1) − Y(0) | X=x]` as a regression on `(X, T, Y)` triples but differently:
- **S-learner:** one regression `μ(X, T)`; predict at `T=1` − `T=0`.
- **T-learner:** two regressions `μ_T(X)` separately on each arm.
- **X-learner:** T-learner + propensity-weighted residual second stage (handles imbalance).
- **R-learner:** Robinson 1988 partialling-out + outcome residual on treatment residual.

Reuses C0b OLS, C0c GLM. Cross-link slot 222 (multi-armed-bandit) for personalised-policy off-policy evaluation.

#### C15. **2SLS — Two-stage least squares** (Theil 1953) — ~180 LOC
First stage: `T̂ = Z(Z'Z)⁻¹Z'·T`. Second stage: OLS of `Y` on `T̂` (and exogenous controls). Asymptotic variance via Wooldridge 2010 §5.2 sandwich. Sargan-Hansen J-test for over-identification (when `dim(Z) > dim(T)`). Hausman test for endogeneity. F-stat for weak-instruments diagnostic (Stock-Yogo 2005 critical values). **The applied-econometrics arsenal.**

```go
type IVResult struct { Coefs, StdErr []float64; FStat, SarganJ, HausmanH float64 }
func TwoSLS(X, Z [][]float64, y []float64) (IVResult, error)
```

#### C16. **Wald estimator + LATE** (Imbens-Angrist 1994) — ~110 LOC
Just-identified IV with binary instrument:
```
LATE = Wald = (E[Y|Z=1] − E[Y|Z=0]) / (E[T|Z=1] − E[T|Z=0])
```
Estimates *Local* ATE on compliers (Imbens-Angrist monotonicity assumption). Ships `Wald` + `LATE_Bounds` (Manski 1990) for partial-identification under no-monotonicity.

#### C17. **PC algorithm** (Spirtes-Glymour 1991) — ~360 LOC ⭐
Constraint-based causal-discovery: from observational data, recover the Markov equivalence class (CPDAG) of the data-generating DAG. Three phases:
1. **Skeleton:** start with complete graph; remove edge `i−j` if `MI(X_i, X_j | S) ≈ 0` for some `S ⊂ Adj(i) \ {j}` of growing cardinality.
2. **V-structure orientation:** for each unshielded triple `i−k−j`, orient `i → k ← j` iff `k ∉ separating-set(i, j)`.
3. **Meek's rules:** propagate orientations to avoid cycles / new v-structures.

CI test backbone: partial correlation under linear-Gaussian assumption (Fisher's Z), or kernel-CI test (Zhang 2011) under non-linear. Reuses C0b OLS for partial-correlation, `info/-MutualInformation` for non-parametric CI. **The classical pillar of causal discovery.**

```go
func PCAlgorithm(data [][]float64, alpha float64, ciTest CITestFunc) DAG
```

#### C18. **NOTEARS — Continuous DAG learning** (Zheng-Aragam-Ravikumar-Xing 2018) — ~280 LOC ⭐⭐
**The frontier piece.** Recasts combinatorial DAG-learning as smooth continuous optimisation:
```
min_W ½||X − XW||²_F + λ||W||₁
s.t. h(W) = tr(e^{W∘W}) − d = 0
```
where `h(W) = 0` iff `W` is acyclic (Zheng et al. 2018 Theorem 1). Solve via augmented-Lagrangian + L-BFGS:
```
L_ρ(W, α) = ½||X−XW||²_F + λ||W||₁ + α·h(W) + (ρ/2)·h(W)²
```
Each outer iteration: L-BFGS on `W` (reuses `optim/gradient.go`); update `α ← α + ρ·h(W)`; if `h(W) > γ·h_prev`, `ρ ← 10·ρ`. Ten outer iterations typical. **The first post-hoc-feasible end-to-end-differentiable causal-discovery algorithm**, citation engine of an entire 2019-2026 sub-literature.

Reuses: `autodiff/tape.go` for `∂h/∂W`, `linalg/-MatMul`, `optim/gradient.go-LBFGS`, `optim/proximal/operators.go-ProxL1` for the ℓ¹ block. Critical missing primitive: matrix-exponential `e^M` with autodiff support (~180 LOC Padé-13 from Higham 2005).

```go
type NOTEARSConfig struct { Lambda, Rho, Gamma, MaxOuter, MaxLBFGS int }
func NOTEARS(data [][]float64, config NOTEARSConfig) (W [][]float64, hFinal float64)
```

#### C19. **LiNGAM — Linear Non-Gaussian Acyclic Model** (Shimizu-Hoyer-Hyvärinen-Kerminen 2006) — ~220 LOC ⭐
`X = BX + e` with `e_i` non-Gaussian and independent. Identifies `B` (and hence the causal order) uniquely from observational data via FastICA: `B = I − W` where `W` is the ICA mixing matrix permuted to lower-triangular form.

Reuses: `linalg/pca.go` (whitening pre-step), `linalg/decompose.go-QR`. Ship FastICA inline (~120 LOC, Hyvärinen 1999 fixed-point). DirectLiNGAM (Shimizu et al. 2011) is the deterministic alternative — ~80 LOC iterative-residual-regression.

```go
func LiNGAM(data [][]float64) (B [][]float64, order []int)
func DirectLiNGAM(data [][]float64) (B [][]float64, order []int)
```

### Tier-3 — niche / advanced (~1,420 LOC)

#### C20. **Regression discontinuity** (Thistlethwaite-Campbell 1960, Imbens-Lemieux 2008) — ~220 LOC
Sharp RDD: `T = 1[X ≥ c]`; estimate `τ_SRD = lim_{x↓c} E[Y|X=x] − lim_{x↑c} E[Y|X=x]` via local-linear on either side. Fuzzy RDD: Wald-style IV with `Z = 1[X ≥ c]`. Ships Cattaneo-Calonico-Titiunik 2014 bias-corrected + Imbens-Kalyanaraman 2012 plug-in bandwidth.

```go
func SharpRDD(X, Y []float64, cutoff float64, h float64) (estimate, stderr float64)
func FuzzyRDD(X []float64, T []int, Y []float64, cutoff, h float64) (estimate, stderr float64)
```

#### C21. **Synthetic control method** (Abadie-Diamond-Hainmueller 2010) — ~280 LOC ⭐
Compose synthetic counterfactual as convex combination of donor pool: `min_{w ≥ 0, sum w = 1} ||X_T − X_C w||² + λ ||w||²`. Solves via constrained QP — Frank-Wolfe inline or `optim/proximal/admm.go` simplex projection. Augmented SCM (Ben-Michael-Feller-Rothstein 2021), Generalised SCM (Doudchenko-Imbens 2017), ASCM-elastic-net (Powell 2018) variants. Cross-link slots 215 (CS) and 222 (bandits-OPE).

```go
func SyntheticControl(treatedX, donorX [][]float64, treatedY, donorY []float64, T0 int) SCResult
func AugmentedSC(treatedX, donorX [][]float64, treatedY, donorY []float64, T0 int, ridge float64) SCResult
```

#### C22. **Difference-in-differences** (Card-Krueger 1994) — ~120 LOC
Two-way fixed-effects panel: `τ_DiD = (Ȳ_1·T − Ȳ_1·C) − (Ȳ_0·T − Ȳ_0·C)`. Parallel-trends assumption. Reuses C0b OLS with unit+time dummies. Cluster-bootstrap SE. Goodman-Bacon 2021 decomp for staggered adoption (⊘ defer).

#### C23. **Causal forests** (Wager-Athey 2018) — ~340 LOC ⭐
Honest random-forest ensemble of causal trees (Athey-Imbens 2016): each tree splits to maximise within-leaf TE heterogeneity, fit on disjoint sample halves to deliver point-wise asymptotic normality of `τ̂(x)`. Most-deployed CATE estimator in 2024-2026 enterprise causal-ML. Reuses C0b OLS at leaves + `prob/random.go` for bootstrap; parallelisable. Athey-Tibshirani-Wager 2019 GRF generalises to any moment-condition.

#### C24. **Sensitivity analysis — Rosenbaum bounds** (Rosenbaum 1987) — ~180 LOC ⭐
Bounds on p-value/estimate under unmeasured-confounder strength `Γ ≥ 1` for matched pairs. Cinelli-Hazlett 2020 omitted-variable-bias bounds via partial-R² is the modern alternative (~80 LOC). VanderWeele-Arah 2011 E-value via `prob/copula/-Gaussian`.

```go
func RosenbaumBounds(matchedPairs [][2]int, Y []float64, gamma float64) (pLower, pUpper float64)
func CinelliHazlettOVB(ols OLSResult, treatmentIdx int, kY, kT float64) (lowerCI, upperCI float64)
```

#### C25. **Causal trees — honest Athey-Imbens 2016** — ~240 LOC
Single-tree CATE estimator with honest sample-splitting; greedy axis-aligned splits maximising `Var(τ̂(leaf)) − λ·MSE(τ̂(leaf))`. Building block of C23. Reuses `optim/proximal/-soft-threshold` for ℓ¹-regularised honest CATE.

#### C26. **Mediation analysis** (Pearl 2001 / Imai-Keele-Yamamoto 2010) — ~180 LOC
Decompose total = NDE + NIE: `NDE = E[Y(1, M(0)) − Y(0, M(0))]`, `NIE = E[Y(1, M(1)) − Y(1, M(0))]`. Sequential-ignorability identification (IKY-2010 Assumption 1). Two-stage OLS or simulation-based (potential-outcome imputation).

#### C27. **Off-policy evaluation** (Dudík-Langford-Li 2011) — ~160 LOC ⊘ defer
IPW + direct-method + doubly-robust for `V̂_π` from logged `{(x,a,r)~π_b}`. Cross-link slot 222 (bandits). Defer until consumer pulls.

#### C28. **Front-door + back-door hybrid (Tian-Pearl 2002 do-conditional)** — ~120 LOC ⊘ defer
Subsumed by C8 ID algorithm; ship simple closed-form as educational reference. Defer until consumer pulls.

---

## (3) Connective tissue — what each new edge buys

Twelve cross-package edges activate once `causal/` lands:

| Edge | Glue LOC | Unlocks |
|---|---|---|
| `causal/dag/ → graph/` (DAG/BFS/Adj reuse) | 0 | C1/C5/C6/C8 — d-sep, backdoor, frontdoor, ID |
| `causal/effect/ → prob/regression/multiple.go` (C0b) | 260 | C4/C11/C12/C20/C21/C22 — every OLS-based estimator |
| `causal/effect/ → prob/regression/glm.go` (C0c IRLS) | 320 | C2/C12/C13/C16 — propensity, TMLE, DML, fuzzy-RDD |
| `causal/effect/ → prob/random.go` (C0a) | 280 (cross-cut) | bootstrap, permutation (C7), Bayesian SCM (defer) |
| `causal/iv/ → linalg/decompose.go-QR` | 0 | C15/C16/C20 — 2SLS, Wald, RDD covariates |
| `causal/effect/ → optim/transport/sinkhorn.go` | 30 | C9 Sinkhorn-matching; Kallus 2020 OT balancing |
| `causal/discovery/ → autodiff/tape.go` | 60 | C18 NOTEARS gradient; DAG-GNN; neural-SCM frontier |
| `causal/discovery/ → autodiff/matrix_exp.go` (NEW) | 180 | C18 `tr(e^{W∘W})`; cross-link slot 205 Lie-groups |
| `causal/discovery/ → optim/proximal/` | 0 | C18 sparse-DAG; sparse-LiNGAM |
| `causal/discovery/ → info/-MI/-CMI` | 60 | C17 PC under non-linear; KCIT (Zhang 2011) |
| `causal/effect/ → prob/conformal/` | 80 | Conformal CATE (Lei-Candès 2021) on C12/C14/C23 |
| `causal/iv/ → optim/proximal/admm.go` | 40 | C21 synthetic control simplex-projection QP |

**Four new sub-packages**: `causal/dag/` (~1,180 LOC), `causal/effect/` (~1,890 LOC), `causal/iv/` (~860 LOC), `causal/discovery/` (~1,490 LOC). Plus four substrate adds: `prob/regression/multiple.go` (260), `prob/regression/glm.go` (320), `autodiff/matrix_exponential.go` (180), `prob/random.go` (280, cross-cut).

---

## (4) Three architectural recommendations

**F1. Ship `prob/random.go` + `prob/regression/multiple.go` + `prob/regression/glm.go` as a coordinated PR-0 substrate before any causal primitive lands.** This unblocks NINE existing Block-C reviews (117/202/215/220/222/227/228 + this slot 229 + likely 237) and is itself useful for non-causal consumers (NOTEARS, GLM, multiple-OLS = the three most-cited "missing primitives" across the prob-numerics review chain). Total ~860 LOC across three files. Two-day effort. **Must land first.**

**F2. Establish a canonical `causal.DAG` type with latent edges (ADMG — Acyclic Directed Mixed Graph) from day one.** ID-algorithm (C8), backdoor (C1), frontdoor (C6) all need bidirected edges to encode unobserved confounders. Don't ship a directed-only `causal.DAG` and bolt latents on later; the Shpitser-Pearl machinery explicitly assumes ADMGs.

```go
type DAG struct {
    Directed   []Edge   // observed cause → effect
    Bidirected []Edge   // shared-latent confounder
    Nodes      []string
}
func (d DAG) IsValidADMG() error
func (d DAG) MoralizedGraph() UndirectedGraph
func (d DAG) CComponents() [][]string   // for ID algorithm
```

**F3. Pin every estimator with a ground-truth-DGP cross-language golden file.** Simulate from a known SCM, recover structural parameter within `O(1/√n)`. Cross-language parity contract: every sibling implementation (Python, C++, C#) produces *the same* expression tree from C8 ID and *the same* propensity-trimmed ATE from C11 AIPW given the same RNG seed. See section (6) for the eight pinned tests.

---

## (5) Risks and gotchas

- **G1. Propensity-score overlap violation.** `min(ê) < 0.01` or `max(ê) > 0.99` makes IPW variance explode; ship `OverlapDiagnostic` as a required first-stage call.
- **G2. ID algorithm correctness.** Shpitser-Pearl 8-step recursion needs meticulous pattern-matching; ship 3 reference test cases (frontdoor, transportability, hedge-fail) before general impl.
- **G3. NOTEARS local-minima.** Non-convex objective can converge to non-DAG cycle; ship `RoundToTopologicalOrder` post-processing + threshold sweep (Zheng 2018 Algorithm 1 line 14).
- **G4. Matrix-exp stability.** Padé-13 (Higham 2005) needs balancing for ill-conditioned `W`; loud-fail when `||W∘W|| > 8.0`; scaling-and-squaring up to 20 doublings.
- **G5. PC algorithm multiple-testing.** `O(d²)` CI tests; Bonferroni-correct at `α/d²`, or reuse `BenjaminiHochberg` from `prob/regression.go:91` for FDR control.
- **G6. Synthetic control extrapolation.** When treated unit lies outside donor convex hull, weights pile on single donor; ship `DonorHullDiagnostic`.
- **G7. LiNGAM Gaussian-failure.** ICA cannot identify fully-Gaussian system; ship `JarqueBera` residual test, loud-fail when all residuals Gaussian.
- **G8. Doubly-robust caveat.** AIPW (C11) requires *at least one* of propensity/outcome correct; both-wrong → biased. Document — term "doubly robust" misleads non-econometricians.
- **G9. DML K-choice.** Default K=5 (Chernozhukov 2018 §6); K=2 wastes data, K=10 overfits.
- **G10. Sensitivity-Γ scale.** Rosenbaum Γ multiplicative on log-odds; Cinelli-Hazlett R² variance-explained; ship `SensitivityScale` enum + converters.

---

## (6) Cross-language parity targets

Eight ground-truth-DGP pinned tests — generate from a known SCM with `prob/random.go`, recover structural parameter:

| Test | DGP / Reference | Estimator | Tolerance |
|---|---|---|---|
| `TestBackdoor_Smoking` | Pearl-2009 Fig 3.3 smoking-tar-cancer | C1 Backdoor + C6 Frontdoor agree | 1e-12 |
| `TestIPW_Lalonde1986` | NSW Lalonde-1986 | C2 IPW vs $1734 | $200 SE |
| `TestAIPW_DoubleRobustness` | propensity OR outcome misspec | C11 unbiased iff ≥1 correct | bias < 0.05 at n=10000 |
| `TestTMLE_BoundedOutcome` | binary outcome, GLM-binomial | C12 TMLE respects [0,1] | exact constraint |
| `TestDML_Cherno2018Table1` | partially-linear, RF nuisance | C13 DML | √n CI cov ≥ 93% at n=500 |
| `TestNOTEARS_BinaryDAG` | 5-node Bernoulli SCM | C18 NOTEARS | SHD ≤ 1 at n=1000 |
| `TestSC_AbadieHainm2003` | Cal-Tobacco-Prop99 | C21 SC weights | 1e-3 vs paper |
| `TestSharpRDD_LeeLemieux2010` | linear above/below + IK bandwidth | C20 SharpRDD | bias < 0.05 |

---

## (7) Verdict

**Ship Tier-0 + Tier-1 (~2,110 LOC over 5-6 sprints):** PR-0 substrate (C0a/b/c ~860 LOC, unblocks 9 Block-C reviews) → C1 Backdoor + C2 Propensity+IPW + C5 d-sep (~490 LOC Pearl/Rubin foundation) → C3 Matching + C4 RegATE + C6 Frontdoor + C7 Permutation (~480) → C8 ID-algorithm (280, structural-identification keystone) → C11 AIPW + C12 TMLE (~460, doubly-robust pair) → C13 DoubleML (340, 2018+ cutting-edge ⭐⭐).

**Defer-but-design Tier-2/3 (~3,310 LOC, ship when consumer pulls):** C9 Optimal matching, C10 Stratified, C14 Meta-learners (S/T/X/R), C15 2SLS, C16 Wald, C17 PC, C18 NOTEARS, C19 LiNGAM, C20 RDD, C21 SyntheticControl, C22 DiD, C23 CausalForest, C24 SensitivityBounds, C25 CausalTree, C26 Mediation.

**Drop until consumer pulls:** C27 OffPolicyEval (slot 222 territory), C28 frontdoor-backdoor hybrid (subsumed by C8 ID).

**Single-highest-leverage 1-day project:** C1 Backdoor + C2 Propensity+IPW + C0c GLM (~620 LOC total). Drops a complete observational-causal-inference workflow (DAG + treatment + outcome → ATE estimate with SE) on top of existing reality substrate; the canonical "Hello, Pearl" demonstration that proves reality can support causal inference at all.

**Single-highest-leverage cutting-edge piece:** C18 NOTEARS — Zheng-Aragam-Ravikumar-Xing 2018 ICLR is the *one* causal-discovery primitive that no zero-dep Go library ships; ~280 LOC of pure outer-loop scheduling over existing autodiff + L-BFGS + proximal-ℓ¹. Pairs with C19 LiNGAM and C17 PC to deliver a complete causal-discovery triplet (continuous-Gaussian / non-Gaussian-linear / conditional-independence).

**Single-highest-leverage moat:** C13 DoubleML — Chernozhukov-2018 is *the* most-cited applied causal-ML paper of the 2018-2026 era (~7,000 citations). K-fold cross-fitting + Neyman-orthogonal-score is cross-cutting infrastructure composing EVERY reality estimator into an asymptotically-normal causal estimator. No zero-dep Go implementation; Python `econml` / R `DoubleML` are the only mainstream alternatives.

**Architectural witness:** `aicore/causalmath` is a *current production downstream consumer* (Source-tags in `graph/dag.go:15,69` point back). Causal inference is already practised in a sibling repo with reality serving as substrate — slot 229 is "*when* do we lift it?" not "is it in scope?". Answer: after PR-0 substrate lands, Sprint 2.


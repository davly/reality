# 336 — dive-bo-acquisition (EI / UCB / PI / KG / PES / MES / qEI / Thompson audit)

## Headline
Reality ships **zero Bayesian-optimisation surface** (no `optim/bayesian.go`, no `gp/`, no acquisition-function file anywhere) — every BO ingredient is queued in three prior reviews (102 T2.21, 169 S18, 222 B17, 237 G36-G41) but **never landed**; the cheapest Day-1 PR is a 4-acquisition-function file (~520 LOC: EI + UCB + PI + Thompson) plus a minimal MaternGP backbone (~280 LOC) that satisfies five prior deferrals (102/169/222/227/237) with one shared math primitive.

## Findings (existing audit)

- **Topic prompt's premise that `optim/bayesian.go` exists at slot 110 is wrong** — `Glob optim/*.go` (verified) returns: `genetic.go gradient.go gradient_validated.go interpolate.go linear.go metaheuristic.go rootfind.go` plus `proximal/` and `transport/` sub-packages — no `bayesian.go`, no `bo.go`, no `acquisition.go`, no `gp.go` anywhere in repo.
- Repo-wide grep on `BayesOpt|BayesianOptimization|ExpectedImprovement|UpperConfidenceBound|ProbabilityOfImprovement|KnowledgeGradient|EntropySearch|MaxValueEntropy|qEI|GIBBON|ThompsonSampl` (excluding gametheory/bandit.go's Beta-Bernoulli `ThompsonSampling`) returns **zero callable BO matches** across all 22 packages.
- `gametheory/bandit.go:88-108` ships `ThompsonSampling(successes, failures []int, rng) int` — Beta-Bernoulli posterior, **discrete arms only**, NOT a continuous-domain GP-Thompson sampler. Fine to keep as-is; rename in BO context to `gp.ThompsonSampleContinuous`.
- Substrate score is paradoxically high. The four pieces every BO acquisition function actually needs are all shipping:
  - `prob/distributions.go:32` `NormalPDF(x, mu, sigma)` — gates EI's `σ φ(z)` term.
  - `prob/distributions.go:47` `NormalCDF(x, mu, sigma)` — gates EI's `(μ−f*)Φ(z)` term and PI's `Φ((μ−f*−ξ)/σ)`.
  - `prob/distributions.go:67` `NormalQuantile(p, mu, sigma)` — gates Bayes-UCB-style β-quantile acquisition.
  - `linalg/decompose.go:266,316` `CholeskyDecompose` + `CholeskySolve` — gate the GP posterior `(K+σ²I)⁻¹` two-solve.
  - `optim/gradient.go:LBFGS` — gates **inner** acquisition-maximisation step (multi-start L-BFGS over α(x)).
  - `infogeo/mmd.go:7,16` `Kernel` interface + `GaussianKernel(bandwidth)` — gates the RBF-kernel GP surrogate.
- Five prior reviews have all named the same primitives, **all deferred, none landed**:
  - **102 T2.21** (`reviews/overnight-400/agents/102-optim-missing.md:109`): "Gaussian Process Bayesian Optimisation (Močkus 1975, Snoek-Larochelle-Adams 2012) — Expected Improvement, Upper Confidence Bound, Probability of Improvement acquisition functions."
  - **102 T3.16** (line 166): "Batch BO (q-EI, Wang et al. NeurIPS 2016) — for parallel evaluation."
  - **169 S18** (`agents/169-synergy-prob-optim.md:116`): "BayesianOptimisation(f, bounds, surrogate, acq, n_init, n_iter) → x_best ~280 LOC" + EI/UCB/PI/TS as 4-line closures.
  - **222 B17** (`agents/222-new-bandits.md:144`): "GP-UCB / IGP-UCB / GP-TS (Srinivas-Krause-Kakade-Seeger 2010; Chowdhury-Gopalan 2017) ~250 LOC … blocked on `prob.GaussianProcessRegression`."
  - **237 G36-G41** (`agents/237-new-gaussian-process.md:166-170`): GP-UCB (90 LOC) + GP-EI (90) + GP-PI (60) + GP-Thompson (80) + GP-KnowledgeGradient (100) = "~420 LOC ship-once" — **owned-here per the slot-237 architectural placement**.
- **227 U25** (per 237's cross-link table): "Bayesian Optimisation … G37-G41 acquisition functions IDENTICAL surface; 90+90+80+80+80 = 420 LOC ship-once" — identical claim, identical primitives, three different agent reports demanding the same code.
- The repo has **no Gaussian-Process anything** (per slot 237 grep: 0/42 GP primitives), so even an ideal acquisition-function file sits in vacuum without a posterior to call. PR-1 must co-ship a minimal `gp.GPPosterior(K, k_*, k_**, σ², y) → (μ, σ²)` (slot 237 G7, ~80 LOC) before the acquisition functions become consumable.

## Acquisition-function taxonomy (research-frontier crosscheck)

| Acquisition | Year | Closed-form? | Citation | LOC |
|-------------|------|:------------:|----------|----:|
| **PI** Probability of Improvement | 1964 | yes (NormalCDF) | Kushner *J. Basic Eng.* 86 | 40 |
| **EI** Expected Improvement | 1978 / 1998 | yes (NormalPDF + NormalCDF) | Močkus 1978; Jones-Schonlau-Welch *J. Global Opt.* 13 | 70 |
| **UCB** GP-UCB | 1992 / 2010 | yes | Cox-John 1992; Srinivas-Krause-Kakade-Seeger *ICML* | 50 |
| **TS** Thompson sampling (continuous) | 1933 / 2014 | sample posterior path | Thompson 1933; Russo-Van Roy *MOR* 39 | 80 |
| **KG** Knowledge Gradient | 2008 | Monte-Carlo (or Gauss-Hermite for 1-D) | Frazier-Powell-Dayanik *SIAM J. Opt.* 19 | 150 |
| **ES** Entropy Search | 2012 | Monte-Carlo | Hennig-Schuler *JMLR* 13 | 350 |
| **PES** Predictive Entropy Search | 2014 | EP + spectral approx | Hernández-Lobato-Hoffman-Ghahramani *NeurIPS* | 450 |
| **MES** Max-value Entropy Search | 2017 | Gumbel-sample max-values | Wang-Jegelka *ICML* | 180 |
| **GIBBON** | 2021 | EP-free MES variant | Moss-Leslie-Rayson *AISTATS* | 220 |
| **qEI** parallel/batch EI | 2010 / 2016 | MC fantasy-then-optimize | Ginsbourger-Le Riche-Carraro 2010; Wang-Hutter-Zoghi-Matheson-de Freitas *NeurIPS* | 200 |
| **qUCB / qKG** | 2017 | MC | Wang-Jegelka; Wu-Frazier *NeurIPS* | 180 |
| **EHVI** (multi-objective EI) | 2003 | closed-form 2-D, MC ≥3-D | Emmerich-Giannakoglou-Naujoks *IEEE TEC* 10 | 300 |
| **AEI / NEI** noisy/augmented EI | 2006 / 2017 | closed-form | Huang-Allen-Notz-Zeng; Letham-Karrer-Ottoni-Bakshy *Bayes. Anal.* | 90 |

## Concrete recommendations

### T0 — there is no T0
Nothing to audit; nothing exists. The topic-prompt's "audit existing EI / PI / UCB" maps to "verify they are absent" (verified: zero matches).

### T1 — Day-1 PR: Backbone GP + EI/UCB/PI/Thompson — ~520 LOC
Single PR landing five primitives in `gp/` (new sub-package matching slot 237 architectural recommendation):

- `gp/posterior.go` — `GPPosterior(X [][]float64, y []float64, k Kernel, sigma2 float64, xStar []float64) (mu, var float64)` and batch-form `GPPosteriorBatch`. Pure composition: `linalg.CholeskyDecompose((K+σ²I), n, L)` + two `linalg.CholeskySolve` calls. ~150 LOC including jitter retry (slot 237 G18).
- `gp/acquisition.go`:
  - `ExpectedImprovement(mu, sigma, fBest, xi float64) float64` — closed-form `(μ−f*−ξ)Φ(z) + σ φ(z)` with `z = (μ−f*−ξ)/σ`; guard `σ < 1e-9 → return 0`. ~30 LOC.
  - `UpperConfidenceBound(mu, sigma, beta float64) float64` — `μ + √β · σ`. ~10 LOC.
  - `ProbabilityOfImprovement(mu, sigma, fBest, xi float64) float64` — `Φ((μ−f*−ξ)/σ)`. ~15 LOC.
  - `ThompsonSampleContinuous(model GPModel, X [][]float64, rng RNG, out []float64)` — sample posterior path on candidate grid via Cholesky-of-posterior-covariance; argmax pick. ~120 LOC (depends on slot 117 RNG keystone — defer if RNG absent and ship as 80 LOC discrete-grid version).
- `gp/optimize.go` — `MaximizeAcquisition(alpha func([]float64) float64, bounds [][2]float64, nStarts int, rng RNG) ([]float64, float64)` multi-start L-BFGS over α(x), wraps existing `optim.LBFGS`. ~100 LOC.
- `gp/bo.go` — `BayesianOptimization(f func([]float64) float64, bounds [][2]float64, k Kernel, sigma2 float64, acq AcquisitionFunc, nInit, nIter int, rng RNG) (xBest, fBest)`. Outer loop: random-init `nInit` points, fit GP, maximise acquisition, evaluate `f`, append, repeat. ~120 LOC.

Saturates **R-EI-MOCKUS-CLOSED-FORM 3/3** (see pin section). Closes 102 T2.21, 169 S18, 222 B17, 227 U25, 237 G36-G39 simultaneously.

### T2 — Knowledge Gradient via Monte Carlo — ~150 LOC
Frazier-Powell-Dayanik 2008. MC estimate of `KG(x) = E[max_x' μ_{n+1}(x' | x, y(x)) − max_x' μ_n(x')]`:
1. Fantasise `m` posterior samples `y_j ~ N(μ_n(x), σ_n²(x))`.
2. For each fantasy, recompute posterior mean (rank-1 Sherman-Morrison update of Cholesky — sibling of LinUCB's update, slot 222 B11).
3. Take max over discretised candidate set, average over fantasies.
Closed-form 1-D KG via Gauss-Hermite quadrature (`m=20` knots) for the cheapest path. Pin against EI in σ→0 limit (KG(x) ≡ EI(x) when only `μ` matters).

### T3 — Max-value Entropy Search (MES) — ~180 LOC
Wang-Jegelka 2017 ICML. Sampling-based information-gain acquisition:
1. Sample `M=10` Gumbel-approximated posterior maxima `f*_m ~ p(f* | D)` via Gumbel-with-CDF-approximation.
2. `MES(x) = (1/M) Σ_m [γ_m φ(γ_m)/(2Φ(γ_m)) − log Φ(γ_m)]` where `γ_m = (f*_m − μ(x))/σ(x)`.
Pin against EI in low-noise/single-Gumbel-sample limit. **Faster than PES (T5) by 10× while matching regret on Branin/Hartmann benchmarks.**

### T4 — qEI parallel batch acquisition — ~200 LOC
Wang-Hutter-Zoghi-Matheson-de Freitas 2016 NeurIPS Marmin/Chevalier-Ginsbourger 2014 closed-form for q≤4.
- **Closed-form** for `q ≤ 4` via Genz multivariate-normal CDF (no shipping yet — see slot 117 missing).
- **Monte-Carlo** for `q ≥ 5`: fantasy `q` candidates jointly, evaluate `EI` of best fantasy. ~120 LOC.
- **Constant Liar** simple baseline (Ginsbourger-Le Riche-Carraro 2010): pick first via EI, lie that `f(x_1) = μ(x_1)`, refit GP, pick next via EI, repeat q times. ~80 LOC, pure greedy.
Pin: `qEI(N=1) ≡ EI` exactly.

### T5 — Predictive Entropy Search (PES) — ~450 LOC (deferred to dedicated PR)
Hernández-Lobato-Hoffman-Ghahramani 2014. Information about `x*` argmax via Expectation Propagation on Gaussian factors. **Genuinely complex** — needs: sampling `x*_m` via Thompson on spectral-approximation (Bochner + RFF, slot 237 G31), per-`x*_m` constrained-GP-posterior via EP (slot 237 G29), entropy difference. Defer until slot 237 G29-EP and G31-RFF land.

### T6 — Generic acquisition optimiser (within-step) — ~250 LOC
The unsung hero. Real-world BO regret is dominated not by acquisition choice but by **inner-loop α(x)-maximisation quality**. Ship three optimisers:
- **Multi-start L-BFGS** (T1's default) — fast, local.
- **DIRECT** (Jones-Perttunen-Stuckman 1993, slot 102 T2.17) — Lipschitz-free deterministic global; matches BoTorch / Spearmint default.
- **CMA-ES** (slot 335 Day-1 deliverable) — covariance-adaptation; the 2016+ default for high-dim BO.
Auto-select by `dim ≤ 6 → DIRECT, dim > 6 → multi-start-L-BFGS, dim > 30 → CMA-ES`.

### T7 — Noisy / Augmented EI — ~90 LOC
Huang-Allen-Notz-Zeng 2006; Letham-Karrer-Ottoni-Bakshy 2017 *Bayesian Analysis*. Replace `f_best` with `μ(x_best)` (the posterior mean at the incumbent) — eliminates the EI-degenerate-at-observed-points pathology when observations are noisy. Closed-form, ~30 LOC delta over T1 EI.

### T8 — GIBBON — ~220 LOC (research-frontier)
Moss-Leslie-Rayson 2021 AISTATS. EP-free MES variant; sometimes labelled "general-purpose information-theoretic". Pin against MES in single-fidelity case — both are lower-bound surrogates of the same MI.

### Day-1 PR recommendation: T1 = 520 LOC + slot 237 G7 backbone (80 LOC) = **~600 LOC**
Closes 102 T2.21, 169 S18, 222 B17, 227 U25, 237 G36-G39 (five prior deferrals, single PR). Adds the **first GP primitive in the entire repo** as a side-effect.

## R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

**R-EI-MOCKUS-CLOSED-FORM 3/3** — three derivations of the same EI value at 1e-12:
1. `ExpectedImprovement(μ, σ, f*, ξ=0)` closed-form `(μ−f*)Φ(z) + σφ(z)`.
2. Trapezoidal-rule numerical integration of `∫ max(0, μ + σz − f*) φ(z) dz` over `z ∈ [-10, 10]` with 10⁵ knots → 1e-9 agreement.
3. Monte-Carlo with 10⁶ samples `y_j ~ N(μ, σ²)`, `EI ≈ (1/M) Σ max(0, y_j − f*)` → 1e-3 agreement (CLT bound). Three orthogonal evaluation paths, single closed-form pin.

**R-UCB-DEGENERATE-LIMIT 3/3** — three regimes of UCB collapse:
1. `UCB(μ, σ, β=0) = μ` exactly (pure exploit).
2. `UCB(μ, σ→0, β=any) → μ` exactly (zero-uncertainty exploit).
3. `UCB(μ, σ, β→∞) → ∞` proportionally to `√β · σ` (pure explore — sortable order matches `σ`-ranking).
Pin all three at machine precision.

**R-qEI-EI-EQUIVALENCE 3/3** — `qEI(q=1) ≡ EI` via three implementations:
1. Closed-form qEI via 1-D Genz integral (degenerate to EI when q=1).
2. Monte-Carlo qEI with `q=1, M=10⁶` samples → 1e-3 EI.
3. Constant-Liar qEI with `q=1` (no lying needed, identity by construction).

**R-KG-EI-LIMIT 1/1** (bonus) — `KG(x) → EI(x)` as posterior std at unobserved points dominates posterior mean uncertainty everywhere else. Pin on a single-point-fit Branin GP.

## Numerical pitfalls

1. **EI degenerates at observed points** when `σ(x_obs) → 0` — closed form gives `0/0`. Guard `σ < 1e-9 → EI = 0`. **AEI/NEI (T7) eliminates this entirely** by replacing `f_best` with `μ(x_best)`.
2. **GP posterior variance non-negativity** — Cholesky+two-solve formula `σ² = k(x*,x*) − k_*ᵀ(K+σ²I)⁻¹k_*` can return tiny-negative values (1e-16) due to FP cancellation; clamp to `max(σ², 1e-12)` before `sqrt`.
3. **UCB β-schedule** — Srinivas et al. 2010 prescribe `β_t = 2 log(t² 2π² /6δ) + 2d log(t² d b r √(log(4dα/δ)))` for theoretical sublinear regret; document and ship Kandasamy 2018's empirical `β_t = 0.2·d·log(2t)` as the practical default.
4. **Acquisition optimiser local-minima trap** — multi-start with `≥10·d` restarts; document as the dominant compute cost.
5. **Gumbel sample for MES** — Wang-Jegelka 2017 use empirical-CDF inversion; FP-stable via Stehlé-Steinle 2014.
6. **qEI fantasy step explosion** — naive qEI is O(q!); enforce greedy fantasy ordering per Ginsbourger 2010.
7. **Noisy-observation EI** — vanilla EI overshoots when observations have noise σ²_obs > 0; switch to AEI/NEI automatically when sigma2 > 1e-6.
8. **Bound box scaling** — always normalise `bounds` to `[0,1]^d` before Cholesky; otherwise lengthscale priors become problem-specific.

## Cross-link map

- **slot 102 T2.21 / T3.16** — EI/UCB/PI + qEI: T1 + T4 closes.
- **slot 169 S18** — `BayesianOptimisation` 280 LOC: T1 IS this primitive (with the GP backbone made explicit).
- **slot 222 B17** — GP-UCB / IGP-UCB / GP-TS: T1 closes B17 directly; B17 was blocked on the missing GP regression — T1 ships it.
- **slot 227 U25** — UQ Bayesian-Optimisation: T1 closes.
- **slot 237 G7 + G36-G40** — explicit ownership; this slot operationalises 237's Tier-1 keystone.
- **slot 263 quasi-MC + slot 264 MLMC** — Sobol-init for BO `nInit` candidates (better than uniform-random).
- **slot 117 prob-missing (RNG/Sample)** — Thompson sampling and qEI Monte-Carlo paths block on it.
- **slot 335 CMA-ES** — T6 acquisition-optimiser-portfolio depends on its landing.

## Consumer pull (why this matters more than its LOC suggests)

- **Hyperparameter optimisation** — Optuna / Ax / SMAC / BoTorch are *all* GP-EI by default. Reality-as-foundation cannot serve any AI-system parent (LimitlessGodfather, Pistachio, etc.) tuning hyperparameters without it.
- **Drug discovery** (every BO benchmark; Janus 2018, BoTorch tutorials).
- **Material-property optimisation** — citation peak 2020-2025 in *npj Comp. Materials*.
- **Neural-architecture search** — DARTS-BO, NAS-Bench-201 use GP-EI baselines.
- **Calibration of expensive simulators** — orbital-mechanics / fluids / CFD calibration consumers.
- **A/B-test multi-armed bandit cousin** — slot 222 B17 IS this surface.
The single most-cited 21st-century optimisation algorithm absent from reality.

## Sources

- `optim/genetic.go`, `optim/gradient.go`, `optim/linear.go`, `optim/metaheuristic.go`, `optim/rootfind.go`, `optim/proximal/`, `optim/transport/` (verified — no `bayesian.go`, no `bo.go`, no acquisition file).
- `prob/distributions.go:32,47,67` — NormalPDF/CDF/Quantile substrate.
- `linalg/decompose.go:266,316` — CholeskyDecompose/Solve substrate.
- `infogeo/mmd.go:7,16` — Kernel + GaussianKernel substrate.
- `gametheory/bandit.go:88-108` — discrete Beta-Bernoulli Thompson (NOT continuous-domain BO).
- `reviews/overnight-400/agents/102-optim-missing.md:109,166` (T2.21, T3.16).
- `reviews/overnight-400/agents/169-synergy-prob-optim.md:116-123,219` (S18 + PR-6).
- `reviews/overnight-400/agents/222-new-bandits.md:144` (B17 GP-UCB).
- `reviews/overnight-400/agents/237-new-gaussian-process.md:166-170,210-211` (G36-G41 acquisition zoo + PR-3).
- Mockus, J. (1978). "On Bayesian methods for seeking the extremum." *Optimization Techniques IFIP Conf.*, Springer LNCS 27.
- Jones, D.R., Schonlau, M., Welch, W.J. (1998). "Efficient global optimization of expensive black-box functions." *J. Global Optimization* 13(4):455-492.
- Kushner, H.J. (1964). "A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise." *J. Basic Eng.* 86(1):97-106.
- Srinivas, N., Krause, A., Kakade, S., Seeger, M. (2010). "Gaussian process optimization in the bandit setting: No regret and experimental design." *ICML*.
- Frazier, P.I., Powell, W.B., Dayanik, S. (2008). "A knowledge gradient policy for sequential information collection." *SIAM J. Control Optim.* 47(5):2410-2439.
- Hennig, P., Schuler, C.J. (2012). "Entropy search for information-efficient global optimization." *JMLR* 13:1809-1837.
- Hernández-Lobato, J.M., Hoffman, M.W., Ghahramani, Z. (2014). "Predictive entropy search for efficient global optimization of black-box functions." *NeurIPS*.
- Wang, Z., Jegelka, S. (2017). "Max-value entropy search for efficient Bayesian optimization." *ICML*.
- Ginsbourger, D., Le Riche, R., Carraro, L. (2010). "Kriging is well-suited to parallelize optimization." *Computational Intelligence in Expensive Optimization Problems*, Springer.
- Wang, J., Clark, S.C., Liu, E., Frazier, P.I. (2016). "Parallel Bayesian global optimization of expensive functions." *NeurIPS* (qKG / qEI).
- Moss, H.B., Leslie, D.S., Rayson, P. (2021). "GIBBON: General-purpose information-based Bayesian optimisation." *J. Machine Learning Research* 22.
- Letham, B., Karrer, B., Ottoni, G., Bakshy, E. (2019). "Constrained Bayesian optimization with noisy experiments." *Bayesian Analysis* 14(2):495-519.
- Wilson, J.T., Hutter, F., Deisenroth, M.P. (2018). "Maximizing acquisition functions for Bayesian optimization." *NeurIPS*.
- BoTorch (Balandat et al. 2020 NeurIPS), GPyOpt (González et al. 2016), Spearmint (Snoek-Larochelle-Adams 2012 NeurIPS) — reference implementations.

# 220 | new-stochastic-opt

**Topic:** Stochastic optimization — SAA, SGD with momentum families, variance reduction (SVRG, SAGA, SAG), adaptive optimizers (Adam, AdamW, RMSprop, Adagrad, Lion), federated/distributed averaging.
**Block:** C (cutting-edge math, what reality is missing). **Date:** 2026-05-08.
**Scope:** the **numerical-optimization** view of stochastic methods — finite-sum minimization, expected-loss / online-streaming, learning-rate schedules, stochastic L-BFGS, distributed / federated SGD. Distinct from 169 (`prob×optim` deterministic fits MLE/EM/MAP/VI/BO) and from 195 (`optim×prob` SDE-as-sampler / SGLD / CMA-ES / IGO / PAC-Bayes). 220 owns the **finite-sum minimizer + ML-optimizer + variance-reduction-for-supervised-learning** axis.

## Two-line summary

`reality/optim/` ships ZERO stochastic-gradient finite-sum machinery — verified by repo-wide grep returning zero matches for any of `SGD|MiniBatch|Momentum|Polyak|Nesterov|NAG|Adam|AdamW|Adagrad|Adadelta|RMSprop|Lion|Lookahead|SVRG|SAGA|SAG|MISO|Katyusha|SDCA|Hogwild|FedAvg|FederatedAveraging|ParameterServer|AllReduce|RingAllReduce|stochasticLBFGS|oLBFGS|importanceSampling|cosineSchedule|cyclical|warmRestart|warmup|EmpiricalRisk|ExpectedRisk|SAA|SampleAverage` across `optim/`, `prob/`, `linalg/`, and any other package — the only RNG-touching surfaces in the entire repo are `optim.SimulatedAnnealing` (Kirkpatrick-Gelatt-Vecchi 1983) + `optim.GeneticAlgorithm` (Holland 1975 with inlined Box-Muller at `genetic.go:58-65`); 169 named the deterministic-fit half (S1-S18 = MLE/MAP/EM/VI/BO) and 195 named the SDE-as-sampler half (N1-N22 = SGLD/SG-HMC/CMA-ES/PAC-Bayes); **220 names the finite-sum-minimizer half — twenty-three primitives F1-F23 totalling ~3,150 LOC of pure connective tissue** that compose existing `optim.GradientDescent` loop body + `optim.LBFGS` two-loop recursion + `optim.SimulatedAnnealing` RNG-acceptance plumbing + `optim/proximal.ProxL1/ProxNonNeg/ProxBox` to deliver the entire 2010s-2020s ML-optimizer stack. Cheapest one-day standalone is **F1 SAA + F2 EmpiricalRiskWrapper + F3 RobbinsMonroSGD + F4 MiniBatchSGD (~480 LOC)** which lands the **finite-sum optimization framework** that gates F5-F23. Highest-leverage one-week unlock is **F8 Adam + F9 AdamW + F10 Adagrad + F11 RMSprop + F12 Lion (~520 LOC)** because they collectively saturate the **R-OPTIMIZER-FAMILY-EQUIVALENCE 5/5** pin (all five reduce to plain SGD when their adaptive denominators degenerate) and unlock every ML-training consumer reality could ever serve. Architectural keystone is the **FiniteSumLoss interface** (~30 LOC) — `func(theta, sampleIdx) (loss, grad)` — co-shipped with 169-S14 `RNGSampler` and 195-N3 unified RNG keystones into a single `optim/stochastic.go` substrate file.

---

## 0. State of play (verified file-walk)

### `optim/` finite-sum / SGD surface = ZERO

Verified absent (repo-wide grep on `optim/`, `prob/`, all sub-packages):

- **No stochastic-gradient anything:** zero matches for `SGD`, `Stochastic`, `MiniBatch`, `Momentum`, `Polyak1964`, `Nesterov`, `NAG`, `accelerated`, `heavyball`, `heavy-ball`, `heavy_ball`.
- **No adaptive optimizers:** zero matches for `Adam`, `AdamW`, `Adagrad`, `Adadelta`, `RMSprop`, `Lion`, `Lookahead`, `Yogi`, `Nadam`, `RAdam`, `LAMB`, `LARS`, `Adafactor`, `Shampoo`, `K-FAC`.
- **No variance reduction:** zero matches for `SVRG`, `SAGA`, `SAG`, `MISO`, `S-MISO`, `Katyusha`, `SDCA`, `Finito`.
- **No stochastic second-order:** zero matches for `oLBFGS`, `online-LBFGS`, `stochastic.LBFGS`, `subsampled.Newton`, `subsampled.Cubic`, `Newton.sketch`, `LiSSA`.
- **No distributed / federated:** zero matches for `Hogwild`, `Hogwild!`, `FedAvg`, `FederatedAveraging`, `parameterServer`, `parameter.server`, `AllReduce`, `RingAllReduce`, `gradient.compression`, `quantized.SGD`, `signSGD`, `LocalSGD`.
- **No learning-rate schedules:** zero matches for `cosineSchedule`, `cosine.schedule`, `cyclicalLR`, `warmRestart`, `linearSchedule`, `stepSchedule`, `exponentialDecay`, `oneCycle`, `triangular`, `polynomialDecay`, `inverseTimeDecay`.
- **No SAA / sample average approximation:** zero matches for `SAA`, `SampleAverage`, `SampleAverageApproximation`.
- **No online learning / regret:** zero matches for `OnlineGradient`, `OGD`, `regret`, `expertAdvice`, `multiplicativeWeights`, `FTRL`, `FTL`, `EXP3`, `Hedge` (174-G7 names `OnlineLearner` interface — still pending).
- **No importance sampling:** zero matches for `importanceSampled.SGD`, `importance.SGD`.
- **No empirical-risk / expected-risk wrappers:** zero matches for `EmpiricalRisk`, `ExpectedRisk`, `ERM`, `Population.Risk`.
- **No epoch / data-iteration abstraction:** zero matches for `Epoch`, `DataLoader`, `Sampler.WithoutReplacement`, `Shuffle.Epoch`.

The only RNG-aware functions in the entire repo (verified by grep `interface{ Float64\(\) float64 }|\*rand\.Rand`):

- `optim/genetic.go:44-178` (`GeneticAlgorithm`, BLX-α + Box-Muller inlined at lines 58-65).
- `optim/metaheuristic.go:38-93` (`SimulatedAnnealing`, Boltzmann acceptance + geometric cooling).
- `prob/markov.go:99-139` (`MarkovSimulate`, hard-coded LCG — exposes no RNG abstraction).

### `prob/` data-streaming surface = ZERO

Verified absent: `Welford`, `running.variance`, `streaming.mean`, `Kahan.summation` (the Kahan compensated-summation we'd want for stable epoch-loss accumulation does not exist in `prob/`). The only loss-related functions are `prob.LogLoss` and `prob.BrierScore` (per `prob.go:48-66`) which are scalar over single (label, prediction) pairs — no batch / mini-batch wrappers, no sample-index iterators, no online-mean accumulators.

### Cross-coupling: zero today (verified by grep both directions)

```
$ grep -r "github.com/davly/reality/optim" prob/ ; echo "---"
$ grep -r "github.com/davly/reality/prob"  optim/
---
(no matches in either direction)
```

This is the same vacuum 169 and 195 named — **220 inherits their architectural-keystone proposals (the unified RNG interface) and adds the finite-sum FiniteSumLoss interface on top.**

---

## 1. The conceptual unlock — finite-sum is the ML-optimizer's fundamental abstraction

Every ML optimization problem is a **finite-sum** of per-sample losses:

```
minimize  F(θ) = (1/N) Σ_{i=1}^N f_i(θ)
```

(or the **expected-loss / population-risk** generalization `F(θ) = E_{ξ~P}[f(θ; ξ)]` when streaming online from a distribution P).

Vanilla `optim.GradientDescent` requires a **full-gradient closure** `grad(θ, g)` that fills `g = ∇F(θ)`. This is an O(N·d) computation per iteration — prohibitive when N = 10⁶ samples. The entire SGD canon (Robbins-Monro 1951 → Adam 2014 → Lion 2023) is built on the observation that an **unbiased stochastic estimator** of `∇F(θ)` — namely `∇f_i(θ)` for `i ~ Uniform(1..N)` — drives convergence to a stationary point under Robbins-Monro step-size conditions (Σ a_k = ∞, Σ a_k² < ∞).

The minimal abstraction that makes the entire SGD canon ship as compositions of `optim.GradientDescent`'s loop body is:

```go
// FiniteSumLoss decomposes an objective F(θ) = (1/N) Σ f_i(θ) into per-sample
// loss + per-sample gradient. Implementations write the gradient ∇f_i(θ) into g.
type FiniteSumLoss interface {
    N() int                                                  // number of samples
    Sample(theta []float64, idx int, g []float64) float64    // ∇f_i, returns f_i
}
```

(or equivalently a closure pair `(N int, sampleGrad func(theta []float64, idx int, g []float64) float64)`).

**This 30-LOC interface gates every primitive in F1-F23.** Once it exists, SGD is `optim.GradientDescent` with `grad(θ, g) ← FiniteSumLoss.Sample(θ, rng.Intn(N), g)` plus a per-step learning-rate schedule. Mini-batch SGD averages over `B` such draws. Adam adds a per-coordinate moment buffer. SVRG adds an outer-epoch snapshot of the full gradient. SAGA adds a per-sample gradient table. **The expansion from interface to algorithm is two to twenty lines for every optimizer in the canon.**

The deeper unifier: Robbins-Monro 1951 + Polyak-Ruppert 1990 + Bottou-Curtis-Nocedal 2018 *Optimization Methods for Large-Scale Machine Learning* (SIAM Rev. 60:223) all frame the entire SGD family as **fixed-point iteration on the population-gradient operator** `T(θ) = θ − a · E[∇f_i(θ)]`. Variance reduction (SVRG/SAGA/SAG) is the same iteration with a *control-variate* baseline that drives `Var[ĝ] → 0`. Adaptive methods (Adam/RMSprop/Adagrad) precondition the iteration by an empirical-Fisher diagonal. Federated averaging is `T(θ)` applied locally on K shards then arithmetic-meaned. **One unifying view; twenty-three deliverables.**

---

## 2. Twenty-three synergy primitives (F1-F23, ~3,150 LOC pure glue)

Numbered ascending by composition-depth. Each lists (capability, composition of existing primitives, LOC).

### Tier 0 — substrate (ships first, gates everything else)

**F1 SampleAverageApproximation(stochasticObj, n_samples, rng) → deterministic_obj** [~110 LOC]
Shapiro-Dentcheva-Ruszczyński 2009 *Lectures on Stochastic Programming* §5 SAA: replace `min_θ E_{ξ~P}[f(θ; ξ)]` with `min_θ (1/n) Σ_{j=1}^n f(θ; ξ_j)` for a fixed Monte-Carlo sample. Composes any user-supplied `f(θ; ξ)` + `optim.RNG` (195-N3) + caller's choice of deterministic optimizer (`optim.LBFGS`, `optim.GradientDescent`). **Two-paragraph wrapper that converts every existing optim solver into a stochastic-programming solver.** Saturation pin: SAA approximation rate `O(1/√n)` for the optimal-value gap (Shapiro 2003, Theorem 5.6) — verifiable on a quadratic stochastic program.

**F2 EmpiricalRiskWrapper(per_sample_loss, X, y) → FiniteSumLoss** [~80 LOC]
Adapter: given user-supplied `per_sample_loss(theta, x_i, y_i, g) → loss` (scalar + gradient via the autodiff sub-package or hand-derived), returns a `FiniteSumLoss` over the dataset (X, y). Vapnik 1991/1998 ERM principle realized as a 2-line constructor. Adapters for the canonical losses ship pre-rolled: `prob.LogLoss` → binary logistic, `prob.BrierScore` → squared-error classification, MSE → linear regression, hinge → SVM.

**F3 RobbinsMonroSGD(loss FiniteSumLoss, theta0, schedule, n_iter, rng) → theta_T** [~140 LOC]
Robbins-Monro 1951 *Annals of Math. Stat.* 22:400 stochastic approximation. `θ_{k+1} = θ_k − a_k · ∇f_{i_k}(θ_k)` with `i_k ~ Uniform(1..N)` and `a_k = a_0 · k^{-α}` for `α ∈ (0.5, 1]`. **Direct overlap with 195-N1; 220 owns the finite-sum / ML-supervised view, 195 owns the SDE-stationary-distribution view; co-ship as a single primitive.**

**F4 MiniBatchSGD(loss FiniteSumLoss, theta0, batch_size, schedule, n_iter, rng) → theta_T** [~150 LOC]
Mini-batch averaging: per step draw `B` indices uniformly without replacement, average their gradients, take a step. Bottou-Curtis-Nocedal 2018 §3.2: variance scales as `O(1/B)`, so wall-clock per-iteration scales as `O(B)` — the optimal `B*` is hardware-dependent (cache, vector lanes), conventionally `B ∈ [32, 512]`. Composes F3 with an inner accumulator loop.

### Tier 1 — momentum families (composes Tier 0)

**F5 HeavyBallMomentum(loss, theta0, lr, beta, n_iter, rng) → theta_T** [~110 LOC]
Polyak 1964 *USSR Comput. Math. Math. Phys.* 4:1 heavy-ball: `v_{k+1} = β v_k + g_k; θ_{k+1} = θ_k − lr · v_{k+1}`. Three-line modification of F4 with a velocity buffer `v`. Achieves accelerated `O(1/√L·μ)` rate on quadratic strongly-convex objectives (faster than vanilla `O(L/μ)`).

**F6 NesterovAcceleratedSGD(loss, theta0, lr, beta, n_iter, rng) → theta_T** [~120 LOC]
Nesterov 1983 *Soviet Math. Dokl.* 27:372 — accelerated gradient with **lookahead**: `θ_{lookahead} = θ_k + β v_k; v_{k+1} = β v_k − lr · ∇f(θ_{lookahead}); θ_{k+1} = θ_k + v_{k+1}`. Sutskever-Martens-Dahl-Hinton 2013 *ICML* identifies the lookahead-vs-plain-momentum subtlety; document both forms (PyTorch convention vs. Sutskever convention) and pin the equivalence.

**F7 PolyakRuppertAveraging(sgd_iterator, n_total) → theta_avg** [~50 LOC]
Polyak-1990 / Ruppert-1988 tail-averaging — direct overlap with 195-N8; co-ship as one primitive. With `lr_k = a/k^α`, `α ∈ (0.5, 1)`, the average `θ̄_T = (1/T) Σ_{k=1}^T θ_k` achieves CLT with **optimal asymptotic variance** equal to inverse-Fisher-information.

### Tier 2 — adaptive optimizers (Adam family, ships ML-canonical stack)

**F8 Adam(loss, theta0, lr, beta1, beta2, eps, n_iter, rng) → theta_T** [~150 LOC]
Kingma-Ba 2014 *ICLR* Adam — adaptive moment estimation:
```
m_k = β_1 m_{k-1} + (1 − β_1) g_k         (1st moment EMA)
v_k = β_2 v_{k-1} + (1 − β_2) g_k²        (2nd moment EMA, elementwise)
m̂_k = m_k / (1 − β_1^k);   v̂_k = v_k / (1 − β_2^k)   (bias correction)
θ_{k+1} = θ_k − lr · m̂_k / (√v̂_k + ε)
```
**The most-cited optimizer in 2010s-2020s ML** (1M+ citations). Per-coordinate moment buffers reuse F3's RNG. Document the Reddi-Kale-Kumar 2018 *ICLR* non-convergence caveat and the AMSGrad correction `v̂_k = max(v̂_{k-1}, v_k)`. Direct overlap with 195-N16; co-ship.

**F9 AdamW(loss, theta0, lr, beta1, beta2, eps, weight_decay, n_iter, rng) → theta_T** [~30 LOC]
Loshchilov-Hutter 2019 *ICLR* — decouples L2 weight-decay from the gradient. `θ_{k+1} = θ_k − lr · (m̂_k / (√v̂_k + ε) + λ θ_k)`. Three-line modification of F8. Document the precision hazard: vanilla "Adam + L2" is **not** Adam-on-MAP-with-Gaussian-prior; the decoupled decay is the mathematically-correct form. The 2019-onwards default for transformer training (BERT/GPT/LLaMA all use AdamW).

**F10 Adagrad(loss, theta0, lr, eps, n_iter, rng) → theta_T** [~80 LOC]
Duchi-Hazan-Singer 2011 *J. Mach. Learn. Res.* 12:2121 — accumulating squared-gradient denominator: `v_k = v_{k-1} + g_k²; θ_{k+1} = θ_k − lr · g_k / (√v_k + ε)`. Per-coordinate adaptive step. Crown-jewel observation: monotonically-shrinking effective LR makes Adagrad the **wrong choice for non-convex problems with shifting curvature** (document and recommend Adam/AdamW).

**F11 RMSprop(loss, theta0, lr, beta, eps, n_iter, rng) → theta_T** [~80 LOC]
Hinton 2012 lecture-notes (cited as Tieleman-Hinton 2012 *Coursera*) — EMA of squared gradient: `v_k = β v_{k-1} + (1-β) g_k²; θ_{k+1} = θ_k − lr · g_k / (√v_k + ε)`. The EMA-not-cumulative-sum fix to Adagrad's monotone-shrinkage problem. Adam = RMSprop + momentum + bias correction. Pin equivalence: at `β_1 = 0` Adam degenerates to RMSprop with bias correction.

**F12 Adadelta(loss, theta0, rho, eps, n_iter, rng) → theta_T** [~110 LOC]
Zeiler 2012 arXiv:1212.5701 — second-order Hessian-free unit-correction: tracks both EMA of gradient² and EMA of update² to make the algorithm scale-free in the gradient. Self-tuning learning rate (no `lr` parameter). Less common in modern practice but historically influential and ships in PyTorch/TF — include for ML-canonical-stack completeness.

**F13 Lion(loss, theta0, lr, beta1, beta2, weight_decay, n_iter, rng) → theta_T** [~100 LOC]
Chen-Liang-Cheng-Hsieh-Hou-Liu-Wei-Yu-Sun-Pham-Sun-Le 2023 *NeurIPS* — EvoLved Sign Momentum: `c_k = β_1 m_{k-1} + (1-β_1) g_k; θ_{k+1} = θ_k − lr · (sign(c_k) + λ θ_k); m_k = β_2 m_{k-1} + (1-β_2) g_k`. **Symbolic-search-discovered optimizer** that matches AdamW with ~2-10× less memory (no v buffer, only m). Default for many 2024+ transformer-training codebases. Saturates the **R-OPTIMIZER-ZOO 5/5** pin (F8/F9/F10/F11/F13 all reduce to plain SGD when their adaptive denominators degenerate).

**F14 Lookahead(inner_optimizer, k, alpha) → theta_T** [~60 LOC]
Zhang-Lucas-Hinton-Ba 2019 *NeurIPS* — wraps any inner optimizer (F8-F13) with a slow-weights buffer: every `k` inner steps, the slow weights are pulled toward the fast weights by `α`. `slow_{k+1} = slow_k + α (fast_k − slow_k); fast_k ← slow_{k+1}` (reset fast). Wrapper, not optimizer; composes any F-family member. **Higher-order primitive** that demonstrates reality's first optimizer-meta-pattern.

### Tier 3 — variance reduction (finite-sum specialty)

**F15 SAG(loss FiniteSumLoss, theta0, lr, n_iter, rng) → theta_T** [~180 LOC]
Roux-Schmidt-Bach 2012 *NIPS* Stochastic Averaged Gradient: maintain a table `g_table[i]` of last-seen `∇f_i(θ_{k_i})` for each i; per step pick i, replace `g_table[i] ← ∇f_i(θ)`, take step `θ ← θ − (lr/N) Σ g_table[j]`. **First linear-rate stochastic method on smooth strongly-convex finite-sums** (`O(ρ^k)` with `ρ < 1`). Storage `O(N·d)`.

**F16 SAGA(loss FiniteSumLoss, theta0, lr, n_iter, rng) → theta_T** [~170 LOC]
Defazio-Bach-Lacoste-Julien 2014 *NIPS* — SAG variant with **unbiased** gradient estimator: `ĝ = ∇f_i(θ) − g_table[i] + (1/N) Σ g_table[j]`. Same `O(N·d)` storage as F15 but cleaner theoretical guarantees (linear rate without the log factor SAG accumulates). **Saturates R-VARIANCE-REDUCTION-FAMILY 3/3:** SAG × SAGA × SVRG all converge at linear rate to within 1e-9 on a strongly-convex quadratic finite-sum benchmark (`f_i(θ) = ½(a_iᵀθ − b_i)² + (μ/2)‖θ‖²` with `μ > 0`). Direct overlap with 195-N15; co-ship.

**F17 SVRG(loss FiniteSumLoss, theta0, lr, epoch_len, n_epoch, rng) → theta_T** [~180 LOC]
Johnson-Zhang 2013 *NIPS* Stochastic Variance Reduced Gradient — epoch-based variance reduction: per epoch compute snapshot `θ̃` and full gradient `μ̃ = (1/N) Σ ∇f_i(θ̃)`; per inner iter `θ_{k+1} = θ_k − lr (∇f_i(θ_k) − ∇f_i(θ̃) + μ̃)`. **Storage `O(d)` only** (no per-sample table) — the SAG/SAGA-vs-SVRG memory-vs-compute tradeoff. Direct overlap with 195-N14; co-ship.

**F18 Katyusha(loss FiniteSumLoss, theta0, lr, tau1, tau2, n_iter, rng) → theta_T** [~240 LOC]
Allen-Zhu 2017 *J. ACM* 64(4) Katyusha — **accelerated variance reduction**, the SVRG analog of Nesterov acceleration. Three-point coupling `x_{k+1} = τ_1 z_k + τ_2 θ̃ + (1-τ_1-τ_2) y_k` with `y, z` momentum buffers. Achieves `O(1/k²)` rate vs SVRG's `O(1/k)` on smooth convex finite-sums; matches the lower bound (Woodworth-Srebro 2016). The single hardest variance-reduction primitive; necessary for the **R-ACCELERATION-LOWER-BOUND** pin.

**F19 SDCA(loss, X, y, lambda, n_iter, rng) → theta_T** [~190 LOC]
Shalev-Shwartz-Zhang 2013 *J. Mach. Learn. Res.* 14(2) Stochastic Dual Coordinate Ascent: solve the dual of `min (1/N) Σ φ_i(xᵢᵀθ) + (λ/2)‖θ‖²` by coordinate ascent on dual variables `α_i`. Linear rate when `φ_i` smooth. **The non-primal variance-reduction approach** — composes the ProxL2 / Fenchel-conjugate machinery from `optim/proximal/` (the Fenchel-conjugate of the squared loss is itself; the Fenchel-conjugate of the logistic loss is the negative entropy of (1-σ(y))). Bridge primitive between `optim/proximal/` and the SDCA family.

### Tier 4 — stochastic second-order (composes Tier 0)

**F20 oLBFGS(loss FiniteSumLoss, theta0, batch, m, n_iter, rng) → theta_T** [~280 LOC]
Schraudolph-Yu-Günter 2007 *AISTATS* online L-BFGS — adapts `optim.LBFGS`'s two-loop recursion with **mini-batch curvature pairs** computed on the same mini-batch as the gradient (otherwise `y_k = g(B_{k+1}) − g(B_k)` is noise-dominated). Composes `optim.LBFGS` two-loop body verbatim with sample-index-aware curvature collection. Bordes-Bottou-Gallinari 2009 SGD-QN refinement. **First primitive in the repo that consumes the existing LBFGS internals** and demonstrates the stochastic-extension pattern.

**F21 SubsampledNewton(loss, theta0, hessian_batch, grad_batch, n_iter, rng) → theta_T** [~220 LOC]
Roosta-Khorasani-Mahoney 2019 *Math. Programming* 174 — sub-sample the Hessian (`|S_H|` rows) **and** the gradient (`|S_g|` rows) per iteration; solve `(H_S + ρ I) d = −g_S` via CG. Composes a CG inner-solver (~80 LOC, missing from repo) + `optim.RNG` for index sampling. **First randomized linear-algebra primitive in `optim/`** (cross-link 213 NLA-randomized SVD/sketch).

**F22 SubsampledCubicRegularization(loss, theta0, sigma, n_iter, rng) → theta_T** [~250 LOC]
Cartis-Gould-Toint 2011 *Math. Programming* 127 ARC + Kohler-Lucchi 2017 *ICML* sub-sampled variant: per iter solve `min_d gᵀd + ½ dᵀHd + (σ/3)‖d‖³`. Cubic regularization gives **dimension-free** rate `O(1/k^{2/3})` on non-convex problems (vs Newton's `O(1/k)`). The 2017+ frontier for non-convex stochastic optim with second-order info. Composes F21's sub-sampled Hessian + a closed-form-cubic-step solver (Conn-Gould-Toint 2000).

### Tier 5 — distributed / federated (orthogonal axis, ships networking-pattern primitives)

**F23 FedAvg(local_optimizer, n_clients, n_local_epochs, n_rounds, weight_fn) → theta_T** [~210 LOC]
McMahan-Moore-Ramage-Hampson-Arcas 2017 *AISTATS* Federated Averaging — per round, K clients each run `n_local_epochs` of their local optimizer (any F8-F19) on their private data shard, then weighted-average their parameters at the server: `θ_global = Σ (n_k/N) θ_k`. **The single-canonical federated-learning primitive** (1B+ deployments via Google Gboard / Apple Siri). Network topology agnostic; this primitive ships the math, not the RPC. Naturally composes any inner optimizer F3-F19 via dependency injection. **Saturates R-FED-CONVERGENCE 3/3:** FedAvg(K=1) ≡ centralized SGD; FedAvg(K=∞, n_local=1) ≡ batch SGD; FedAvg with K shards drawn IID from the same distribution converges at the same `O(1/√(K·T))` rate as centralized SGD with `K·T` samples (Karimireddy-Kale-Mohri-Reddi-Stich-Suresh 2020 *ICML* SCAFFOLD analysis).

**Deferred to v2 (named but not budgeted):**
- **Hogwild!** (Niu-Recht-Re-Wright 2011 *NIPS*) — lock-free async SGD; requires shared-memory primitives (sync.Atomic) which is a Go-runtime concern, not a math concern; defer to consumer-side (aicore service-level).
- **Parameter Server / AllReduce / RingAllReduce** (Li-Andersen-Park-Smola-Ahmed-Josifovski-Long-Shekita-Su 2014 *OSDI*; Sergeev-DelBalso 2018 Horovod) — networking primitives, not math primitives; defer to consumer (service infrastructure).
- **SCAFFOLD / FedProx** (Karimireddy 2020; Li-Sahu-Zaheer-Sanjabi-Talwalkar-Smith 2020) — federated variance-reduction; ships after F16 SAGA + F23 FedAvg if consumer demand materializes.
- **Gradient Compression / signSGD / quantized SGD** (Bernstein-Wang-Azizzadenesheli-Anandkumar 2018; Alistarh-Grubic-Li-Tomioka-Vojnovic 2017) — defer; specialized for distributed-training bandwidth concerns.

### Tier 6 — learning-rate schedules + online convex (cross-cutting)

These are **wrappers, not optimizers** — they ship as additive 30-50 LOC each on top of F3 RobbinsMonroSGD's per-step `a_k` parameter:

- **Step / Exponential / Polynomial / InverseTime decay** (~30 LOC each): `a_k = a_0 · γ^⌊k/step⌋ / a_0 · exp(−γk) / a_0 · (1 + γk)^{-α}`.
- **Linear warmup → cosine annealing** (~50 LOC): Loshchilov-Hutter 2017 *ICLR* SGDR warm-restart `a_k = (a_min + a_max)/2 + (a_max - a_min)/2 · cos(π k/T)`. The single-most-used 2020+ ML schedule.
- **Cyclical LR** (~40 LOC): Smith 2017 *WACV* triangular cyclical schedule; useful for super-convergence and finding optimal `a_max`.
- **OneCycle** (~50 LOC): Smith-Topin 2018 — cosine-warmup-then-cosine-decay; default for fastai.
- **Cosine warm restarts** (~40 LOC): Loshchilov-Hutter 2017 SGDR — cosine schedule that restarts every `T_i` iterations with `T_i = 2 T_{i-1}`.

**Total schedule LOC: ~250.** Ship in a single `optim/schedules.go` file; each is one functional closure `func(k int) float64`.

**Online Convex Optimization (174 cross-link):**
- **OGD (Online Gradient Descent)** (Zinkevich 2003 *ICML*) — F3 with `a_k = O(1/√k)` and adversarial loss-of-the-day; regret `O(√T)`. Exists in 174-G7 OnlineLearner interface; co-ship.
- **FTRL (Follow-the-Regularized-Leader)** (McMahan 2011 *AISTATS* / Shalev-Shwartz-Singer 2007) — proximal-step over cumulative gradient. ~120 LOC; composes `optim/proximal.ProxL1` (the FTRL-Proximal McMahan 2013 *KDD* variant for sparse logistic regression at Google scale).
- **EXP3 / Hedge** (Auer-Cesa-Bianchi-Freund-Schapire 2002) — ships in 174-Cluster-D.

**Importance Sampling SGD** — Zhao-Zhang 2015 / Needell-Srebro-Ward 2014 — direct overlap with 195-N19; co-ship.

---

## 3. Composition graph (DAG)

```
FiniteSumLoss interface (~30 LOC)            ┐
RNG interface (169-S14 + 195-N3 + 220 keystone, ~80 LOC)   ├── architectural keystones
                                              ┘
F1 SAA  ──── deterministic-optimizer wrapper ────── independent
F2 EmpiricalRiskWrapper ── builds FiniteSumLoss from per-sample loss

F3 RobbinsMonroSGD (= 195-N1)
 ├── F4  MiniBatchSGD                                             ┐
 ├── F5  HeavyBallMomentum                                        │
 ├── F6  NesterovAcceleratedSGD                                   │── momentum
 ├── F7  PolyakRuppertAveraging (= 195-N8)                        │   family
 │                                                                ┘
 ├── F8  Adam (= 195-N16)                                         ┐
 ├── F9  AdamW (= 195-N17)                                        │
 ├── F10 Adagrad (= 195-N18a)                                     │── adaptive
 ├── F11 RMSprop (= 195-N18b)                                     │   family
 ├── F12 Adadelta                                                 │
 ├── F13 Lion                                                     │
 ├── F14 Lookahead (wraps F8-F13)                                 │
 │                                                                ┘
 ├── F15 SAG                                                      ┐
 ├── F16 SAGA (= 195-N15)                                         │
 ├── F17 SVRG (= 195-N14)                                         │── variance
 ├── F18 Katyusha                                                 │   reduction
 ├── F19 SDCA                                                     │   family
 │                                                                ┘
 ├── F20 oLBFGS                                                   ┐
 ├── F21 SubsampledNewton (needs CG inner solver)                 │── stochastic
 ├── F22 SubsampledCubicRegularization                            │   second-order
 │                                                                ┘
 └── F23 FedAvg (wraps F3-F19, dependency-injected)               ── distributed

Schedules (~250 LOC) ─── orthogonal wrappers; compose with F3-F19
OGD/FTRL ─── 174-G7 OnlineLearner cross-link
ImportanceSampling SGD (= 195-N19) ─── co-shipped with 195
```

---

## 4. Saturation pins this review unlocks

Per the recent saturation pattern (audio-onset 3-detector 6a55bb4, copula×autodiff 365368a, NGramDice 85a80db):

- **R-OPTIMIZER-FAMILY-EQUIVALENCE 5/5 (F8-F13):** all five adaptive optimizers reduce to plain SGD when their adaptive denominators degenerate (Adam β_2 → 0, RMSprop β → 0, Adagrad after one step, Lion sign(g) ≡ g for scalar problems). Five parallel reductions to one fixed point. Pin: 5 optimizers run on a 1D quadratic with `lr = 1e-3, β_anything = 0` agree to 1e-12 with vanilla SGD trajectory.
- **R-VARIANCE-REDUCTION-FAMILY 3/3 (F15-F17):** SAG × SAGA × SVRG converge at linear rate to within 1e-9 on the same strongly-convex finite-sum benchmark (`f_i(θ) = ½(a_iᵀθ − b_i)² + (μ/2)‖θ‖²`). Three orthogonal storage / unbiasedness profiles converging to the same optimum is the canonical R-MUTUAL idiom.
- **R-ACCELERATION-LOWER-BOUND 2/2 (F6, F18):** Nesterov SGD on smooth convex finite-sum and Katyusha on the same problem both achieve `O(1/k²)` rate and **match the Woodworth-Srebro 2016 lower bound** within a constant factor. Pin demonstrates reality has saturated optimal first-order acceleration.
- **R-MOMENTUM-EQUIVALENCE 3/3 (F5, F6, F8):** heavy-ball momentum × Nesterov × Adam-with-`β_1` agree to 1e-6 on a 100D convex quadratic when their momentum coefficients are matched (β_HB = β_Nesterov = β_1_Adam = 0.9) and adaptive denominators are clamped to 1.
- **R-SAA-CONVERGENCE-RATE 1/1:** F1 SAA with `n` samples on a stochastic quadratic program achieves `O(1/√n)` optimal-value-gap per Shapiro 2003 Theorem 5.6. Pin: gap shrinks at rate `0.51 ± 0.05` across `n ∈ {10, 100, 1000, 10000}`.
- **R-FED-CONVERGENCE-CENTRALIZED-EQUIVALENCE 3/3 (F23):** FedAvg(K=1) ≡ centralized; FedAvg(K=∞, local_epochs=1) ≡ batch; FedAvg(K shards IID) at `O(1/√(K·T))` matches centralized SGD with K·T samples per Karimireddy 2020 SCAFFOLD analysis. **First federated-learning pin in the repo.**

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| (keystone) | FiniteSumLoss interface | 30 | 0 | — |
| (keystone) | RNG interface (= 169-S14 = 195-N3) | 80 | 0 | — |
| F1 | SampleAverageApproximation | 110 | 0 | RNG |
| F2 | EmpiricalRiskWrapper | 80 | 0 | FiniteSumLoss |
| F3 | RobbinsMonroSGD (= 195-N1) | 140 | 0 | FiniteSumLoss, RNG |
| F4 | MiniBatchSGD | 150 | 0 | F3 |
| F5 | HeavyBallMomentum | 110 | 1 | F3 |
| F6 | NesterovAcceleratedSGD | 120 | 1 | F3 |
| F7 | PolyakRuppertAveraging (= 195-N8) | 50 | 1 | F3 |
| F8 | Adam (= 195-N16) | 150 | 2 | F3 |
| F9 | AdamW (= 195-N17) | 30 | 2 | F8 |
| F10 | Adagrad (= 195-N18a) | 80 | 2 | F3 |
| F11 | RMSprop (= 195-N18b) | 80 | 2 | F3 |
| F12 | Adadelta | 110 | 2 | F3 |
| F13 | Lion | 100 | 2 | F3 |
| F14 | Lookahead | 60 | 2 | F8-F13 |
| F15 | SAG | 180 | 3 | F3 |
| F16 | SAGA (= 195-N15) | 170 | 3 | F3 |
| F17 | SVRG (= 195-N14) | 180 | 3 | F3 |
| F18 | Katyusha | 240 | 3 | F17, F6 |
| F19 | SDCA | 190 | 3 | optim/proximal |
| F20 | oLBFGS | 280 | 4 | optim.LBFGS |
| F21 | SubsampledNewton | 220 | 4 | CG inner solver (new ~80 LOC) |
| F22 | SubsampledCubicRegularization | 250 | 4 | F21 |
| F23 | FedAvg | 210 | 5 | F3-F19 (any inner) |
| (sched) | LR schedules (cosine/cyclical/warmup/onecycle/exp/poly/step/inverseTime) | 250 | 6 | F3 |
| (online) | OGD + FTRL (cross-link 174-G7) | 180 | 6 | F3, optim/proximal |
| (importance) | Importance sampling SGD (= 195-N19) | 130 | 6 | F3 |
| **Σ** | | **3,910** | | |

(Excluding the deduplicated overlap with 169 + 195: **net new LOC for 220 ≈ 3,150** since F3, F7, F8, F9, F10, F11, F15, F16, F17, F19-N variants would be co-shipped under either review heading — 220 owns the **finite-sum / ML-optimizer naming + FiniteSumLoss interface + Tier-3 variance reduction + Tier-4 stochastic second-order + Tier-5 federated** ; 195 owns the **SDE-as-sampler naming + SGLD/SG-HMC/CMA-ES/IGO/PAC-Bayes**.)

Pure-glue ratio: ~75 % composition over `optim.GradientDescent` body + `optim.LBFGS` two-loop + `optim/proximal.ProxL1/ProxBox/ProxL2Ball` + Box-Muller from `genetic.go`. ~25 % genuinely-new math (Katyusha three-point coupling at ~120 LOC; SDCA primal-dual at ~100 LOC; Cubic-regularization closed-form step at ~80 LOC; FedAvg correctness proofs at ~60 LOC).

---

## 6. Recommended PR sequence

**PR-1: substrate (FiniteSumLoss + RNG keystones + F1 SAA + F2 ERMWrapper + F3 SGD + F4 MiniBatchSGD) — ~590 LOC source, ~280 LOC tests, single day**
Lands the **finite-sum optimization framework**. Co-ships with 169-S14 PriorSamples and 195-N1 RobbinsMonroSGD as a single architectural-keystone PR (the unified RNG interface lands once, all three reviews' Tier-1 unblocks). **First stochastic-gradient anything in the repo.** Saturates R-SAA-CONVERGENCE-RATE 1/1 + R-MOMENTUM-EQUIVALENCE 3/3 trivial cases.

**PR-2: F5 HeavyBall + F6 Nesterov + F7 PolyakRuppertAveraging — ~280 LOC source, ~140 LOC tests, half-day**
Momentum family. Pin: F5 + F6 agree to 1e-6 on a 100D convex quadratic at matched β.

**PR-3: F8 Adam + F9 AdamW + F10 Adagrad + F11 RMSprop + F12 Adadelta + F13 Lion + F14 Lookahead — ~660 LOC source, ~340 LOC tests, two days**
**The ML-canonical-stack PR.** Lands every adaptive optimizer reality could ever need. Saturates R-OPTIMIZER-FAMILY-EQUIVALENCE 5/5 on a 1D quadratic. Document the Reddi-Kale-Kumar 2018 non-convergence caveat for vanilla Adam and the AMSGrad correction; document the Loshchilov-Hutter 2019 weight-decay-decoupling for AdamW; document the Chen 2023 sign-momentum derivation for Lion. **The single highest-utility PR for downstream ML consumers** (aicore training would import this directly).

**PR-4: F15 SAG + F16 SAGA + F17 SVRG — ~530 LOC source, ~280 LOC tests, two days**
Variance-reduced family. Pin R-VARIANCE-REDUCTION-FAMILY 3/3 on the strongly-convex quadratic finite-sum.

**PR-5: F18 Katyusha + F19 SDCA — ~430 LOC source, ~220 LOC tests, two days**
Accelerated VR + dual coordinate ascent. Pin R-ACCELERATION-LOWER-BOUND 2/2. SDCA composes `optim/proximal.ProxL2`.

**PR-6: F20 oLBFGS + F21 SubsampledNewton + F22 SubsampledCubic — ~750 LOC source, ~360 LOC tests, three days**
Stochastic second-order. F20 reuses `optim.LBFGS` two-loop body; F21 needs a new CG inner solver (~80 LOC). Cross-link to randomized-NLA agent 213.

**PR-7: F23 FedAvg + LR schedules (cosine/cyclical/warmup/onecycle) + OGD + FTRL — ~640 LOC source, ~280 LOC tests, two days**
Federated + schedules + online-convex. **First federated-learning primitive in reality.** Pin R-FED-CONVERGENCE-CENTRALIZED-EQUIVALENCE 3/3. LR schedules ship as functional closures. OGD + FTRL cross-link 174-G7.

Total: ~3,880 LOC source + ~1,900 LOC tests across 7 PRs over ~12 engineer-days. PR-1 is single-day standalone with maximum gating-leverage; PR-3 (Adam-family) is the single highest-utility PR for downstream consumers; PR-7 (FedAvg) is the crown-jewel composition.

---

## 7. Cycle-hazard analysis

Proposed import directions:

```
optim/  ──→  optim/proximal/         (F19 SDCA, FTRL-Proximal)  — already exists for consumer test
optim/  ──→  optim/{gradient,LBFGS}  (F20 oLBFGS reuses LBFGS internals)  — same package
optim/  ──→  prob/                   (NONE — F-series stays optim-internal; per-sample-loss is caller-supplied)
prob/   ──→  optim/                  (NONE — opposite direction reserved for 169 fit primitives)
```

**No new cross-package edges required.** F1-F23 ship entirely within `optim/` (likely as `optim/stochastic.go`, `optim/momentum.go`, `optim/adaptive.go`, `optim/varreduce.go`, `optim/secondorder.go`, `optim/federated.go`, `optim/schedules.go`). The FiniteSumLoss interface and RNG interface are package-internal types in `optim/`.

**Crucially: 220 does NOT need `prob/` for any primitive in F1-F23.** Per-sample loss + gradient is caller-supplied (the `FiniteSumLoss.Sample` closure). Users who want logistic regression import `prob.LogLoss` themselves and wrap it via F2 EmpiricalRiskWrapper. This preserves the same DAG rule as 195-N4 (logp is caller-supplied).

**DAG remains cycle-free.** Verified by enumeration. The interface lives in `optim/`, the consumers are external.

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **F1 SAA:** sample-average-approximation rate `O(1/√n)` for the optimal-value gap requires sub-Gaussian tails on `f(θ; ξ)` (Shapiro 2003 Theorem 5.6 hypothesis); document the failure mode for heavy-tailed losses (use Catoni-Giulini 2017 robust-mean estimator instead).
- **F3 RobbinsMonroSGD:** schedule `α ∈ (0.5, 1]` strictly; `α = 1` is asymptotically slow but correct, `α < 0.5` is divergent. Panic at `α ≤ 0.5`. **Direct lift from 195-N1.**
- **F4 MiniBatchSGD:** without-replacement sampling within an epoch reduces variance by `(N-B)/(N-1)` (finite-population correction, Johnson-Roosta 2020); document but don't auto-enforce — many ML codebases use with-replacement for simplicity.
- **F5 HeavyBall:** Polyak's accelerated rate `O(1/√(L/μ))` requires **strongly convex** objectives; on non-convex problems heavy-ball can oscillate or diverge (Lessard-Recht-Packard 2016 IQC analysis). Document; recommend Nesterov (F6) or Adam (F8) for non-convex.
- **F6 Nesterov:** **two equivalent forms** in the literature (PyTorch convention vs. Sutskever-Martens-Dahl-Hinton 2013 *ICML* convention). Document both and the equivalence transform; pin equivalence test.
- **F8 Adam:** Reddi-Kale-Kumar 2018 *ICLR* — vanilla Adam can fail to converge on certain non-convex problems (Theorem 3 in their paper); document AMSGrad correction `v̂_k = max(v̂_{k-1}, v_k)` as a safer drop-in. Bias-correction terms must use `^k` not `^(k-1)` (off-by-one footgun caught by Goh 2017 *Distill* "Why Momentum Really Works").
- **F9 AdamW:** weight-decay coefficient `λ` is **not** the same scale as the L2 regularization coefficient — re-tune when migrating from Adam+L2 to AdamW. Document the scale relation `λ_AdamW = λ_L2 / lr` (approximately).
- **F10 Adagrad:** per-coordinate `√v_k` denominator monotonically grows; effective step size monotonically shrinks. **Adagrad is the wrong choice for non-convex problems with shifting curvature**; document and recommend Adam/AdamW.
- **F11 RMSprop:** `β` close to 1 (e.g., 0.999) gives stable estimates but slow adaptation; `β ∈ [0.9, 0.99]` typical. Document the `eps` floor (`1e-8`) prevents `g/√v` blow-up.
- **F13 Lion:** `sign()` is **bias-introducing** at zero (sign(0) = 0 vs. distribution mean 0); document the heuristic `sign(0) = 0` convention used in original paper. Lion's effective step size is `|lr|` per coordinate (not `|lr · g|`) — re-tune when migrating from Adam (typically `lr_Lion ≈ 0.1 · lr_Adam`).
- **F15 SAG:** biased gradient estimator; the rate constant has a `log` factor SAGA cleans up. Storage `O(N·d)` — for `N = 10⁶, d = 10⁴` this is 80 GB. Document the practical limit `N·d ≤ 10⁹` (`8 GB`).
- **F16 SAGA:** unbiased; preferred over SAG. Same `O(N·d)` storage hazard as F15.
- **F17 SVRG:** snapshot `θ̃` must be updated every `m = 2N` inner iterations on average per Johnson-Zhang 2013; longer epochs lose variance reduction (drift), shorter epochs lose linear convergence (over-counting full-gradient cost).
- **F18 Katyusha:** parameters `τ_1, τ_2` must satisfy Allen-Zhu 2017 §3 constraints `τ_1 + τ_2 ≤ 1` + `τ_1 ∈ [0, 1/2]` for the contraction proof. Document the canonical values `τ_1 = 1/(L·m)`, `τ_2 = 1/2`.
- **F20 oLBFGS:** curvature pair `(s_k, y_k)` must use the **same mini-batch** for `g(B)` at both `θ_k` and `θ_{k+1}` (Schraudolph-Yu-Günter 2007 §3). Otherwise `y_k` is noise-dominated and the L-BFGS Hessian approximation diverges.
- **F23 FedAvg:** **non-IID client data** breaks the SCAFFOLD-style convergence proof (Karimireddy 2020 *ICML* §4); document client drift and recommend SCAFFOLD or FedProx (deferred to v2). Number of local epochs `E` trades communication-rounds for client drift; `E = 1` is "minibatch SGD" (no drift), `E → ∞` saturates client drift.
- **Schedules:** cosine annealing's `T_max` should be set to total training steps; warm restarts double `T_i`. Document Loshchilov-Hutter 2017 *ICLR* default `T_0 = 100, T_mult = 2`.

---

## 9. Distinct from prior agents (provenance)

- **011-015 autodiff** — names "stochastic AD (reparam + score-function)" Tier-3 but does NOT compose with the SGD outer-loop; this synergy is the consumer-side pull through `optim/` that justifies that autodiff Tier-3 (the per-sample gradient `∇f_i(θ)` is what reverse-mode AD computes when applied to `f_i`).
- **101-105 optim isolation** — 101 names "Adam/AdamW/RMSprop/Lion/Lookahead/SVRG/SAGA/SAG missing"; 102 names "trust-region/Newton-CG/HVP missing"; **220 fills exactly those 101+102 gaps** with the ML-supervised-learning frame. Net new vs 101: F1 SAA, F18 Katyusha, F19 SDCA, F22 SubsampledCubic, F23 FedAvg, all schedules, OGD, FTRL.
- **116-120 prob isolation** — 117 names `RNG/Sample/LogPDF` debt; 220 inherits the 169-S14 + 195-N3 RNG keystone.
- **151-152 synergy-signal-prob, 153 synergy-prob-infogeo, 155 synergy-crypto-prob, 161 synergy-control-prob, 162 synergy-graph-prob** — all orthogonal.
- **163 synergy-optim-autodiff** — names `A12 Adam`, `A18 SVRG` from autodiff-side; 220 ships the same primitives from optim-side. **The two reviews compose: 163 ships autodiff, 220 consumes it.**
- **164 synergy-orbital-optim, 165 synergy-sequence-prob, 166 synergy-acoustics-signal, 167 synergy-audio-signal, 168 synergy-physics-autodiff** — orthogonal.
- **169 synergy-prob-optim (PARTIAL OVERLAP)** — 169 owns the **deterministic fit-distribution-to-data** axis (MLE/MAP/EM/VI/BO, 18 primitives). **220 owns the finite-sum-minimizer axis**. Disjoint primitive rosters by design: 169 has S1-S18 (none stochastic-gradient finite-sum), 220 has F1-F23 (all stochastic-gradient finite-sum). Shared base: both consume the unified RNG interface; both reference `optim/proximal/`. Co-ship the keystone PR.
- **174 synergy-gametheory-optim** — names `G7 OnlineLearner` interface, OGD, no-regret bounds; **220 cross-links** OGD + FTRL into the same `OnlineLearner` interface pattern. Co-ship if PR-7 lands the schedule + online primitives in the same patch.
- **170-194 prior synergies** — orthogonal except where noted above.
- **195 synergy-optim-prob (PARTIAL OVERLAP, MAJOR)** — 195 owns the **SDE-as-sampler / stochastic-approximation / CMA-ES / IGO / PAC-Bayes** axis (22 primitives). **220 owns the finite-sum-minimizer / ML-optimizer / variance-reduction-for-supervised-learning axis**. Mostly-disjoint primitive rosters — overlap is on F3 (= N1), F7 (= N8), F8 (= N16), F9 (= N17), F10 (= N18a), F11 (= N18b), F15 (= ?), F16 (= N15), F17 (= N14). For the overlap set: **co-ship under both review headings, ship code once.** 195 emphasizes the SDE / sampler interpretation; 220 emphasizes the finite-sum / ML-loss interpretation. Net new in 220 vs 195: F1 SAA (stochastic programming, not SDE), F2 ERMWrapper (ML-canonical), F4 MiniBatchSGD, F5 HeavyBall, F6 Nesterov, F12 Adadelta, F13 Lion, F14 Lookahead, F18 Katyusha, F19 SDCA, F20 oLBFGS, F22 SubsampledCubic, F23 FedAvg, all schedules, OGD, FTRL — i.e., 14 of 23 primitives are unique to this slot; 9 of 23 are shared with 195 and ship once.
- **196-219 prior synergies** — orthogonal (color-info, acoustics-fluids, physics-optim, graph-info, topology-signal, chaos-control, fluids-control, prob-changepoint, em-geometry, color-prob, graph-topology, changepoint-timeseries, queue-prob, gametheory-optim, zkmark-crypto, color-prob, geometry-optim, control-optim, em-fluids, topology-signal, chaos-control, fluids-control, prob-changepoint, em-geometry, color-info, acoustics-fluids, physics-optim, graph-info, free-prob, rough-paths, mean-field-games).

**14 of 23 primitives unique to this slot.** 220 is the ML-optimizer-canonical slot in the 400-sequence — every primitive named here is what 2026 ML training stacks call by these specific names (Adam, AdamW, Lion, Lookahead, SVRG, SAGA, FedAvg) rather than the SDE-as-sampler aliases 195 gives them (SGLD, SGHMC, CMA-ES, IGO).

---

## 10. Bottom line

`reality/optim/` ships ZERO finite-sum / mini-batch / momentum / adaptive / variance-reduced / federated primitives despite being the obvious target package for the entire 2010s-2020s ML-optimizer canon. **Twenty-three primitives F1-F23 totalling ~3,910 LOC of pure connective tissue** stand up the entire SGD-with-momentum-families + variance-reduction-for-finite-sums + stochastic-second-order + federated-averaging pipeline on existing v0.10.0 surfaces (specifically: `optim.GradientDescent` loop body, `optim.LBFGS` two-loop recursion, `optim.SimulatedAnnealing` RNG plumbing, `optim/proximal.ProxL1/L2/Box/NonNeg`, Box-Muller from `genetic.go:58-65`).

Cheapest single-day standalone is **PR-1 (substrate + F1 SAA + F2 ERMWrapper + F3 SGD + F4 MiniBatchSGD = 590 LOC)** which lands the **first finite-sum / stochastic-gradient anything** in `optim/` and gates F5-F23 simultaneously. Single highest-utility PR for downstream consumers is **PR-3 (F8 Adam + F9 AdamW + F10 Adagrad + F11 RMSprop + F12 Adadelta + F13 Lion + F14 Lookahead = 660 LOC)** — the canonical ML-optimizer stack that aicore training would import directly. Single hardest is **PR-6 (F20 oLBFGS + F21 SubsampledNewton + F22 SubsampledCubic = 750 LOC)** — stochastic second-order with Hessian sub-sampling. Crown-jewel composition is **PR-7 (F23 FedAvg + schedules + OGD/FTRL = 640 LOC)** — first federated-learning primitive in reality, plus the cross-cutting LR-schedule wrappers and the online-convex bridge to 174-G7.

**Reality is unusually well-positioned for this synergy because (i) `optim.SimulatedAnnealing` already takes an `*math/rand.Rand` parameter (precedent for the unified RNG interface keystone); (ii) `optim.GeneticAlgorithm`'s `interface{Float64() float64}` is the parametric-RNG precedent; (iii) `optim/genetic.go:58-65` already inlines Box-Muller (precedent for any Gaussian-sampling consumer); (iv) `optim/proximal/` already ships `ProxL1/Box/NonNeg/L2Ball/Simplex` (the prox operators FTRL-Proximal and SDCA need); (v) `optim.LBFGS` already implements the two-loop recursion oLBFGS reuses verbatim; (vi) the consumer-side-placement rule (per agents 158-219) recommends F1-F23 in `optim/` (already RNG-aware) rather than a new sub-package — minimum architectural perturbation, maximum ML-optimizer unlock.**

The single most important conceptual identity 220 pins: **Robbins-Monro 1951 SA + Polyak 1964 heavy-ball + Nesterov 1983 acceleration + Duchi-Hazan-Singer 2011 Adagrad + Kingma-Ba 2014 Adam + Loshchilov-Hutter 2019 AdamW + Chen 2023 Lion are seven specializations of the same fixed-point iteration `θ_{k+1} = θ_k − P_k(g_k, θ_{<k})` on the population-gradient operator** — varying only in (a) how `P_k` is computed (identity, momentum, EMA-of-squared, sign), (b) how `g_k` is estimated (single-sample, mini-batch, with/without variance reduction), and (c) how the per-step `lr_k` is scheduled (Robbins-Monro, cosine, cyclical, warm-restart). **R-OPTIMIZER-FAMILY-EQUIVALENCE 7/7 saturation lands when PR-3 ships.** Reality should pin this identity as the single canonical entry point for the entire ML-optimizer literature.

# 222 | new-bandits

**Topic:** Multi-armed bandits — UCB family, Thompson family, EXP3 family, contextual / linear / kernel / GP, best-arm identification, restless / sleeping / mortal / dueling / risk-aware / federated / non-stationary variants. **Block:** C (cutting-edge math, what reality is missing). **Date:** 2026-05-08.
**Scope:** the **bandit canonical surface**, deepened — distinct from 174 (G9 = EXP3 named, no other bandits), 220 (zero bandits, owns finite-sum SGD), 221 (O18 = LinUCB + O25 = EXP3 named, but only the slice the OCO regret-bound framework demands; 221 explicitly defers the BAI / GP-UCB / Lipschitz / restless / dueling / Whittle / mortal / sleeping / federated / risk-aware / Best-of-Both-Worlds twenty primitives to **this** review). 222 owns the **stochastic-bandit + adversarial-bandit + pure-exploration + contextual + structured-arm + non-stationary + preference-feedback + constrained / risk-aware** axes.

## Two-line summary

`reality/gametheory/bandit.go` ships exactly **THREE state-less bandit functions** — `UCB1` (Auer-Cesa-Bianchi-Fischer 2002 at line 37-66), `ThompsonSampling` (Thompson 1933 / Marsaglia-Tsang gamma at line 88-108 with the only RNG-touching Beta sampler in the entire repo at lines 113-176), `EpsilonGreedy` (Sutton-Barto §2 at line 195-221) — all three are **pure functions** taking caller-owned `counts []int`, `rewards []float64` arrays and returning one arm index per call (no cumulative state, no regret tracking, no `Bandit.Update(arm, reward)` interface, no contextual feature vector, no posterior persistence between calls); 174-G9 named the missing `OnlineLearner` interface and 221-O18/O25 named LinUCB/EXP3 as compositions of it — **222 names the remaining 28 bandit primitives B1-B28 totalling ~3,400 LOC of pure connective tissue** covering the four-Pinsker-bound stochastic family (UCB1/UCB-V/KL-UCB/Bayes-UCB), the three adversarial estimators (EXP3/EXP3.P/EXP3-IX/Tsallis-INF), the four BAI track-and-stop variants (succession-elimination/lil'UCB/LUCB/track-and-stop), the four contextual ridge-based learners (LinUCB-disjoint/LinUCB-hybrid/LinTS/OFUL), the GP-UCB / kernelUCB / IGP-UCB Bayesian-optimisation cluster (a direct line of code from `prob.GaussianProcessRegression` which 169-S15 lands), the structured-arm cluster (Lipschitz / Zooming / HOO / combinatorial-CUCB), the non-stationary / restless / sleeping / mortal / switching cluster (Whittle index / D-UCB / SW-UCB / Master-Slave Ada-ILTCB / Rexp3), the preference-feedback cluster (Interleave-Filter / RUCB / DTS / Plackett-Luce-MED), and the BOB / risk-aware / federated / constrained corner. Cheapest one-day standalone is **B1 stochastic-Bandit interface + B2 UCB-V + B3 KL-UCB + B4 BayesUCB (~480 LOC)** which lands the **first stateful bandit anything** and saturates **R-PINSKER-DUAL 4/4** (UCB1 / UCB-V / KL-UCB / BayesUCB all match the Lai-Robbins Σ Δ_i / KL_inf(μ_i, μ*) lower bound on Bernoulli arms — different constants, identical asymptotic rate). Highest-leverage one-week unlock is **B11 LinUCB + B12 LinTS + B13 OFUL + B17 GP-UCB + B23 RUCB-Dueling (~1,050 LOC)** because it collectively closes the contextual + Bayesian-optimisation + preference-feedback gap that gates every modern recommender / hyperparameter-search / RLHF-prompt-selection consumer reality could ever serve. Architectural keystone is the **`StochasticBandit` interface** (~50 LOC, stateful: `Pull() int`, `Update(arm int, reward float64)`, `Regret(true_means []float64) float64`) co-shipped with 221's `OnlineLearner` and 220's `RNGSampler` keystones — three interfaces, one cross-package substrate.

---

## 0. State of play (verified file-walk)

### `gametheory/bandit.go` — three state-less functions, ~222 LOC

Verified by direct read:

- `UCB1(counts []int, rewards []float64, totalPulls int) int` (lines 37-66) — caller owns state, Auer-Cesa-Bianchi-Fischer 2002 formula `x̄_i + √(2 ln N / n_i)`. Returns first 0-count arm if any, else argmax UCB. **No regret tracking.**
- `ThompsonSampling(successes, failures []int, rng) int` (lines 88-108) — Beta-Bernoulli posterior, Marsaglia-Tsang gamma + Box-Muller normal at lines 113-176 (the **only** Beta sampler in the entire repo).
- `EpsilonGreedy(rewards, counts, epsilon, rng) int` (lines 195-221) — random with prob ε, else exploit argmax.

Verified ABSENT (repo-wide grep on every package — overlap with 221 §0 confirmed):

- **No stateful bandit anything:** zero matches for `BanditState`, `MultiArmBandit`, `StochasticBandit`, `BanditAlgorithm`, `Pull`, `Update.*reward`, `Regret`, `cumulativeRegret`, `pseudoRegret`, `expectedRegret`.
- **No KL-divergence bandits:** zero matches for `KLUCB`, `kl-UCB`, `KullbackLeiblerUCB`, `KL_inf`, `Garivier`, `Cappé`, `Lai.Robbins`.
- **No variance-aware UCB:** zero matches for `UCBV`, `UCB-V`, `UCB.V`, `varianceAwareUCB`, `Audibert.Munos.Szepesvari`.
- **No Bayes-UCB:** zero matches for `BayesUCB`, `BayesianUCB`, `Kaufmann.Cappé.Garivier`.
- **No adversarial bandit:** zero matches for `EXP3`, `EXP3P`, `EXP3.P`, `EXP3IX`, `EXP3.IX`, `EXP4`, `Tsallis`, `TsallisINF`, `INF`, `OnlineMirrorDescentBandit`.
- **No best-arm identification:** zero matches for `BestArm`, `BAI`, `successionElimination`, `successiveElimination`, `racing`, `pureExploration`, `lilUCB`, `lil.UCB`, `LUCB`, `LUCB1`, `LUCB++`, `TrackAndStop`, `gao.kveton`, `kalyanakrishnan`.
- **No contextual bandit:** zero matches for `LinUCB`, `linearBandit`, `LinTS`, `OFUL`, `disjointLinUCB`, `hybridLinUCB`, `contextualBandit`, `linearContextual`, `Abbasi.Yadkori.Pal.Szepesvari`, `Li.Chu.Langford.Schapire`, `Agrawal.Goyal`.
- **No Lipschitz / structured / kernel bandit:** zero matches for `LipschitzBandit`, `zoomingTree`, `HOO`, `gaussianProcess.bandit`, `GPUCB`, `GP-UCB`, `IGP-UCB`, `kernelUCB`, `Srinivas.Krause.Kakade.Seeger`, `Combes`, `Bubeck.Munos.Stoltz`.
- **No combinatorial bandit:** zero matches for `combinatorialBandit`, `CUCB`, `CombUCB`, `KvetonWenAshkanValkoEriksson`, `semiBandit`, `LinearSemiBandit`.
- **No restless / Whittle / Markov bandit:** zero matches for `WhittleIndex`, `restlessBandit`, `markovBandit`, `Gittins`, `gittinsIndex`, `WeberWeiss`.
- **No sleeping / mortal:** zero matches for `sleepingBandit`, `mortalBandit`, `Kleinberg.Niculescu.Mizil.Sharma`, `Chakrabarti.Kumar.Radlinski.Upfal`.
- **No dueling / preference / Plackett-Luce:** zero matches for `duelingBandit`, `interleaveFilter`, `BeatTheMean`, `RUCB`, `DTS`, `MergeRUCB`, `Plackett.Luce`, `condorcetWinner`.
- **No BOB / best-of-both-worlds:** zero matches for `BOB`, `bestOfBothWorlds`, `Bubeck.Slivkins`, `Seldin.Slivkins`, `Tsallis.INF.BOB`.
- **No federated bandit:** zero matches for `federatedBandit`, `FedUCB`, `Shi.Shen`, `Wang.Hu.Chen`.
- **No constrained / risk-aware bandit:** zero matches for `constrainedBandit`, `safeBandit`, `CVaR.UCB`, `riskAware`, `meanVarianceBandit`, `KaganLaiYang`, `Sani.Lazaric.Munos`.
- **No non-stationary / switching bandit:** zero matches for `nonStationary`, `switchingBandit`, `D-UCB`, `discountedUCB`, `SW-UCB`, `slidingWindowUCB`, `Rexp3`, `Garivier.Moulines`, `Auer.Gajane.Ortner`.
- **No bandit-feedback RL:** zero matches for `banditFeedback`, `policyGradient.bandit`, `REINFORCE`, `Wu.Wang.Liu`.

**The entire repo's bandit surface is the three pure functions in `bandit.go` plus the inlined Beta sampler.**

### Cross-coupling: zero (verified)

```
$ grep -r "github.com/davly/reality/(linalg|prob|optim)" gametheory/  ; echo "---"
$ grep -r "github.com/davly/reality/gametheory" linalg/ prob/ optim/
---
(no matches in either direction)
```

LinUCB / LinTS / OFUL / GP-UCB **all need `linalg.MatrixSolve` (Sherman-Morrison rank-1 update) + `linalg.Cholesky`** — neither is currently imported. KL-UCB needs `prob.KLDivergence` (binary-Bernoulli special case at `prob/info.go`). GP-UCB needs `prob.GaussianProcessRegression` (169-S15, currently absent). **222's primitives are the cross-edges that motivate 169 + 174 + 221's keystones to land first.**

---

## 1. The conceptual unlock — the three feedback regimes IS the taxonomy

A bandit problem is fully determined by **what the learner observes after each pull**:

| Feedback | What's observed | Algorithm family | Lower bound |
|---|---|---|---|
| **Full information** (all-arm payoff) | full vector `r ∈ ℝ^K` every round | Hedge / OMD-experts (174-G5, 221-O11) | `Ω(√(T log K))` |
| **Bandit / partial** (chosen-arm only) | `r_{a_t}` only; iid → stochastic, arbitrary → adversarial | UCB / Thompson (stoch.) / EXP3 (adv.) | `Ω(√(K T))` (adv) / `Ω(Σ_i log T / Δ_i)` (stoch) |
| **Preference / dueling** | sign of `r_{a_t} − r_{b_t}` (pair) | RUCB / DTS / IF (B23-B25) | `Ω(K log T / Δ²)` |

Within stochastic-bandit, the **complexity is governed entirely by KL-divergence** between arm reward distributions, not raw mean-gap. Lai-Robbins (1985) lower bound:

```
liminf_T R_T / log T  ≥  Σ_{i: μ_i < μ*}  Δ_i / KL_inf(μ_i, μ*)
```

where `KL_inf(μ, μ*) = inf_{ν: E[ν] ≥ μ*} KL(F_μ‖ν)`. **The four "modern" stochastic algorithms (UCB1, UCB-V, KL-UCB, Bayes-UCB) all match this bound up to constants** — but only KL-UCB and Bayes-UCB match it **with the right constant** (i.e., 1, not 8). UCB1 is a slack relaxation via Pinsker's inequality (`KL(p, q) ≥ 2(p−q)²`), which is why UCB1 has an extra factor in the regret bound that vanishes only as `Δ → 0`.

The genius of the framework: **EXP3 is Hedge-with-importance-weighting on the bandit-feedback estimator** (the `r̂_i = r_i · 𝟙[a_t = i] / p_i` Horvitz-Thompson estimator); **LinUCB is UCB1 in feature space with `n_i = trace(A^{-1} x_t x_t^T)` replacing scalar `n_i`**; **GP-UCB is LinUCB in RKHS with the kernel matrix replacing `A`**. One framework subsumes 90% of the canon. The remaining 10% (BAI, dueling, restless, mortal) are the regimes where **the lower-bound shape changes** — pure exploration drops the `log T` factor entirely, dueling loses access to absolute rewards, and restless adds a state-evolution Markov layer.

The four bound classes:

| Regime | Optimal regret | Algorithm |
|---|---|---|
| Stochastic iid (Bernoulli/Gaussian/sub-Gaussian) | `Σ_i Δ_i log T / KL_inf` | KL-UCB, Thompson, Bayes-UCB |
| Adversarial (worst-case) | `√(KT log K)` | EXP3, Tsallis-INF |
| Pure exploration / BAI | `O((H_1 / δ²) log(1/δ))` (sample complexity) | Successive Elimination, Track-and-Stop |
| Contextual linear (`d`-dim) | `√(d T log T)` | LinUCB, OFUL, LinTS |
| Lipschitz `1`-D | `T^{(d_z+1)/(d_z+2)}` (zooming dim `d_z`) | Zooming, HOO |
| GP / RKHS | `√(T γ_T)` (info gain `γ_T`) | GP-UCB, IGP-UCB |
| Restless / Whittle | per-episode `O(K √T)` | Whittle index policy |
| Dueling | `√(K T log T) / Δ` | RUCB, DTS, IF |

---

## 2. Twenty-eight synergy primitives (B1-B28, ~3,400 LOC pure glue)

### Cluster A — stateful stochastic-bandit substrate (B1-B6, ~620 LOC)

**B1. `StochasticBandit` interface** (~50 LOC, the keystone abstraction). Place in `gametheory/bandit_state.go`. Mirrors 174-G4 / 221's `OnlineLearner`:

```go
type StochasticBandit interface {
    Pull() int                                   // arm index for this round
    Update(arm int, reward float64)              // observed reward
    PullCount(arm int) int                       // n_i
    EmpiricalMean(arm int) float64               // x̄_i
    Regret(trueMeans []float64) float64          // pseudo-regret Σ Δ_{a_t}
}
```

Every B2-B10 primitive embeds `BanditState` (counts + sums + sums-of-squares + cumulative-pseudo-regret) and exposes `Pull / Update / Regret`. **The current `bandit.go:UCB1/Thompson/EpsilonGreedy` continue to compile unchanged** as the **stateless adapters** (call `Pull` on a freshly-constructed bandit each call). **PR-1 ships B1-B6 in ~620 LOC plus a 60-LOC `bandit_test.go` golden-file pin against four canonical Bernoulli-arm scenarios.**

**B2. UCB-V** (Audibert-Munos-Szepesvári 2009) (~70 LOC). `UCB(i) = x̄_i + √(2 σ̂_i² ln T / n_i) + 3b ln T / n_i` where `σ̂_i²` is empirical variance (Welford, 220-F6). Tighter than UCB1 when arms have low variance. Saturates **R-VARIANCE-AWARE-CONCENTRATION 2/2** (UCB-V × Bernstein bound on Bernoulli reduces to KL-UCB constant for low-variance arms). Reference: Audibert-Munos-Szepesvári (2009) *Theoretical Computer Science* 410(19).

**B3. KL-UCB** (Garivier-Cappé 2011) (~110 LOC). `UCB(i) = sup{q ∈ [0,1] : n_i · KL(x̄_i, q) ≤ ln T + c ln ln T}`. Inner sup is solved by **bisection on the convex KL-univariate-target** — composes `optim.BisectionMethod` directly. **Asymptotically optimal** (Lai-Robbins constant 1). Bernoulli specialisation uses `prob.KLDivergence` (composed). Saturates **R-PINSKER-DUAL 4/4** (UCB1 / UCB-V / KL-UCB / BayesUCB all match Lai-Robbins on common Bernoulli benchmark, KL-UCB tightest). Reference: Garivier-Cappé (2011) *COLT*.

**B4. Bayes-UCB** (Kaufmann-Cappé-Garivier 2012) (~70 LOC). `UCB(i) = Q_{Beta(α_i, β_i)}(1 − 1/(t (log T)^c))`. Composes the existing Beta-sampler `bandit.go:113-176` plus Beta-quantile via Newton-Raphson on incomplete-beta (use `optim.NewtonRaphson`). Asymptotically optimal. Reference: Kaufmann-Cappé-Garivier (2012) *AISTATS*.

**B5. Gaussian Thompson** (Honda-Takemura 2014) (~80 LOC). Generalises B4 to Gaussian arms with unknown variance: posterior is Normal-Inverse-Gamma, sample is Student-t. Composes `prob.StudentT.Sample` (currently absent — landed by 220-F19 RNG keystone or co-shipped here). Bernoulli + Gaussian co-pinned **R-CONJUGATE-FAMILY 2/2** against B4. Reference: Honda-Takemura (2014) *JMLR*.

**B6. Gittins index** (Gittins 1979) (~240 LOC). The **provably optimal** Bayesian discounted-reward policy for **multi-armed Markov decision processes**. Computed by **Whittle's restart-in-state formulation** + **Katehakis-Veinott** fast iteration. Bernoulli case has closed-form lookup table. References: Gittins (1979) *J. Royal Stat. Soc. B*; Gittins-Glazebrook-Weber (2011) *Multi-Armed Bandit Allocation Indices*. Bridges B1-B5 (myopic) and B26 (restless / Whittle).

### Cluster B — adversarial-bandit (B7-B10, ~340 LOC)

**B7. EXP3** (Auer-Cesa-Bianchi-Freund-Schapire 2002) (~80 LOC). Exponential-weights with importance-weighted reward estimator: `r̂_{i,t} = r_{a_t,t} · 𝟙[i=a_t] / p_{i,t}`, `w_{i,t+1} = w_{i,t} exp(η r̂_{i,t})`. Already named at 174-G9 / 221-O25 — this entry is the **canonical placement under `gametheory/`** rather than under `optim/online/`. Saturates **R-FULL-VS-BANDIT-FEEDBACK 1/1** (EXP3 vs Hedge: identical update, only the estimator differs; cumulative regret blows up by `√K`). Reference: Auer et al. (2002) *SIAM J. Comput.* 32(1):48-77.

**B8. EXP3.P** (high-probability EXP3) (~80 LOC). EXP3 with extra exploration term `γ/K` and adjusted estimator `r̂_{i,t} = (r_{a_t,t} · 𝟙[i=a_t] + β) / p_{i,t}`. Achieves `O(√(KT log K))` with high probability (vs in-expectation for B7). Reference: Auer et al. (2002) §6.

**B9. EXP3-IX** (Neu 2015) (~70 LOC). Implicit-exploration variant: `r̂_{i,t} = r_{a_t,t} · 𝟙[i=a_t] / (p_{i,t} + γ)`. **Stronger high-prob bound than EXP3.P** with simpler analysis. Reference: Neu (2015) *NeurIPS*.

**B10. Tsallis-INF** (Audibert-Bubeck 2009; Zimmert-Seldin 2019) (~110 LOC). Online-Mirror-Descent on the bandit estimator with `1/2`-Tsallis-entropy regulariser instead of negentropy. **Best-of-both-worlds:** matches `O(√(KT))` adversarial regret AND Lai-Robbins stochastic regret simultaneously **without knowing the regime in advance**. The single algorithm that resolves the BOB problem (Bubeck-Slivkins 2012). Saturates **R-BOB 1/1**. Reference: Zimmert-Seldin (2019) *AISTATS*.

### Cluster C — contextual / linear / GP bandits (B11-B17, ~960 LOC)

**B11. LinUCB-disjoint** (Li-Chu-Langford-Schapire 2010) (~120 LOC). Each arm `i` maintains `A_i ∈ ℝ^{d×d}` (regularised Gram), `b_i ∈ ℝ^d`. Pulled arm `a_t = argmax_i x_t^T A_i^{-1} b_i + α √(x_t^T A_i^{-1} x_t)`. **Sherman-Morrison rank-1 update** of `A_i^{-1}` after each pull — composes `linalg.MatrixVectorProduct` + the rank-1-Woodbury identity (currently absent — co-shipped with linalg-side updaters in PR-2). Reference: Li-Chu-Langford-Schapire (2010) *WWW*.

**B12. LinUCB-hybrid** (~80 LOC). Per-arm features `x_{i,t}` PLUS shared user-features `z_t` PLUS per-arm-user interaction `x_{i,t} ⊗ z_t`. Yahoo's actual production article-recommender pattern. Composes B11 plus a shared `A_0` matrix.

**B13. LinTS / Linear Thompson** (Agrawal-Goyal 2013) (~110 LOC). Same `A, b` as B11, but instead of UCB pull a **multivariate-Gaussian sample** `θ̃ ~ N(A^{-1} b, σ² A^{-1})` and pull `argmax x_t^T θ̃`. Composes `linalg.Cholesky` + B5 standard-normal sampler. Empirically dominates LinUCB. Reference: Agrawal-Goyal (2013) *ICML*.

**B14. OFUL** (Abbasi-Yadkori-Pál-Szepesvári 2011) (~150 LOC). Tighter LinUCB confidence width: `α_t = R √(d ln((1 + tL²/λ)/δ)) + √λ S` with explicit dependence on noise-bound `R`, parameter-bound `S`, regularisation `λ`. **First sublinear regret bound for linear contextual bandits with self-normalised concentration.** Saturates **R-SELF-NORMALISED-MARTINGALE 1/1** (de la Peña-Klass-Lai 2009 + OFUL combine into the modern linear-bandit confidence interval). Reference: Abbasi-Yadkori-Pál-Szepesvári (2011) *NeurIPS*.

**B15. SquareCB** (Foster-Rakhlin 2020) (~140 LOC). Reduce contextual-bandit to **online regression** (any oracle): pulled-arm distribution `p_i ∝ 1/(K + γ(r̂* − r̂_i))`, where `r̂_i` is the regression-oracle prediction. **First contextual-bandit algorithm with regret matching the oracle's regression error**, model-free. Composes 220-F8 / 221-O14 ONS as the regression oracle. Reference: Foster-Rakhlin (2020) *ICML*.

**B16. NeuralUCB / NeuralTS** (Zhou-Li-Gu 2020) (~110 LOC, requires autodiff). LinUCB with a `d`-dim NTK-feature representation extracted from a 2-layer neural network. Composes `autodiff.Tape` + `linalg.MatrixVectorProduct`. Out-of-Tier-1 unless autodiff package lands NTK-feature extraction first. Reference: Zhou-Li-Gu (2020) *ICML*.

**B17. GP-UCB / IGP-UCB / GP-TS** (Srinivas-Krause-Kakade-Seeger 2010; Chowdhury-Gopalan 2017) (~250 LOC). Continuum-arm bandit on `[0,1]^d` with Gaussian-process posterior. Pull `x_t = argmax μ_t(x) + β_t σ_t(x)`. Regret bound `O(√(T γ_T))` where `γ_T` is the maximum information gain (kernel-dependent: linear `γ_T = O(d log T)`, RBF `γ_T = O((log T)^{d+1})`, Matérn-ν `γ_T = O(T^{d/(2ν+d)} log T)`). **Composes 169-S15 `prob.GaussianProcessRegression`** which is currently absent. The argmax over `[0,1]^d` is solved by `optim.LBFGS` with random restarts. Reference: Srinivas-Krause-Kakade-Seeger (2010) *ICML*; Chowdhury-Gopalan (2017) *ICML* (improved bound).

### Cluster D — best-arm identification / pure exploration (B18-B21, ~480 LOC)

**B18. Successive Elimination** (Even-Dar-Mannor-Mansour 2002) (~80 LOC). Pull each surviving arm in round-robin, drop arms whose UCB is below another arm's LCB. PAC `(ε, δ)` algorithm: outputs an `ε`-best arm with prob ≥ `1−δ`. Sample complexity `O(Σ_i (1/Δ_i²) log(K/(δ Δ_i)))`. Reference: Even-Dar-Mannor-Mansour (2002) *COLT*.

**B19. lil'UCB** (Jamieson-Malloy-Nowak-Bubeck 2014) (~110 LOC). Uses **law-of-iterated-logarithm** confidence widths: `√((1+β)(1+√ε) · (2σ²(1+ε) ln(ln((1+ε) n_i)/δ)) / n_i)`. Achieves the tight `O(Σ_i log log(1/Δ_i) / Δ_i²)` sample complexity. Reference: Jamieson-Malloy-Nowak-Bubeck (2014) *COLT*.

**B20. LUCB / LUCB++** (Kalyanakrishnan-Tewari-Auer-Stone 2012; Simchowitz-Jamieson 2017) (~100 LOC). Always pulls **two** arms per round: best empirical arm + arm with highest UCB among the rest. Exact-best-arm with optimal sample complexity in the fixed-confidence regime. Reference: Kalyanakrishnan et al. (2012) *ICML*.

**B21. Track-and-Stop** (Garivier-Kaufmann 2016) (~190 LOC). Asymptotically-optimal `(ε=0, δ)` BAI: at each round, plug current empirical means into the Lai-Robbins **lower-bound LP** to obtain optimal allocation `w*(μ̂)`, pull whichever arm is most under-allocated. Stop when `Z(t) > log((t (K-1) + 1)/δ)`. Inner LP composes `optim.SimplexMethod`. Saturates **R-LOWER-BOUND-MATCHING 1/1** for fixed-confidence BAI. Reference: Garivier-Kaufmann (2016) *COLT*.

### Cluster E — structured-arm bandits (B22-B25, ~500 LOC)

**B22. Lipschitz / Zooming bandit** (Kleinberg-Slivkins-Upfal 2008) (~140 LOC). Covers `[0,1]^d` with a hierarchical zooming-tree, only refines high-UCB regions. Regret `T^{(d_z+1)/(d_z+2)}` where `d_z ≤ d` is the zooming dimension. Composes a `geometry.QuadTree` (currently absent — co-ship). Reference: Kleinberg-Slivkins-Upfal (2008) *STOC*.

**B23. HOO — Hierarchical Optimistic Optimisation** (Bubeck-Munos-Stoltz-Szepesvári 2011) (~120 LOC). Refinement of B22 with weakest-link bandit at each tree node. Reference: Bubeck-Munos-Stoltz-Szepesvári (2011) *JMLR*.

**B24. CombUCB / CUCB — combinatorial bandit** (Chen-Wang-Yuan 2013; Kveton-Wen-Ashkan-Valko-Eriksson 2014) (~110 LOC). Action is a **subset / matroid base / path** of size `m`; reward is sum of selected arms' (semibandit) or function thereof (general). UCB on each arm; choose action `S* = argmax Σ_{i∈S} UCB_i`. Inner argmax is matroid-base-greedy or shortest-path — composes `graph.Dijkstra` for the path case. Reference: Kveton et al. (2014) *AISTATS*.

**B25. RUCB / DTS — Dueling bandits** (Zoghi-Whiteson-Munos-de Rijke 2014; Wu-Liu 2016) (~130 LOC). Preference-feedback: each round play arms `(a, b)`, observe `Pr[a beats b] = p_{ab}`. RUCB maintains UCB on the preference-matrix entries `p̂_{ab} + √(α ln t / N_{ab})` and pulls **Condorcet-winner-candidate vs hardest-to-beat-challenger**. DTS = dueling Thompson with Beta posterior on each `p_{ab}`. Saturates **R-PREFERENCE-FEEDBACK 2/2** (RUCB × DTS). Reference: Zoghi et al. (2014) *ICML*.

### Cluster F — non-stationary / restless / sleeping / mortal / federated / risk-aware (B26-B28, ~500 LOC)

**B26. Whittle index policy / Restless bandit** (Whittle 1988; Weber-Weiss 1990) (~200 LOC). Each arm is a 2-state-or-more Markov chain that **evolves whether or not it is pulled**. Whittle index `λ_i(s) = inf{m : passive ≥ active}` is the per-state restart cost making active and passive equally desirable. Pull arms with highest current-state index. **Asymptotically optimal under indexability** (Weber-Weiss). LP-relaxation indexability test composes `optim.SimplexMethod`. Reference: Whittle (1988) *J. Applied Probability*; Weber-Weiss (1990).

**B27. D-UCB / SW-UCB / Rexp3 — non-stationary bandits** (Garivier-Moulines 2011; Auer-Gajane-Ortner 2019) (~150 LOC). Three drop-in replacements for B1-B7 under non-stationary rewards: D-UCB discounts past rewards by `γ^{t-s}`, SW-UCB only uses last `τ` observations, Rexp3 periodically resets EXP3 weights. Detects-and-restarts variant uses CUSUM (`changepoint.CUSUM` already in repo). Reference: Garivier-Moulines (2011) *ALT*.

**B28. Sleeping / Mortal / Federated / Constrained / Risk-aware bandits** (~150 LOC, five 30-LOC sub-primitives). Composition of B1-B11 with side-rules:

- **Sleeping** (Kleinberg-Niculescu-Mizil-Sharma 2010): only a subset of arms is "awake" each round; UCB1-S pulls the highest-UCB awake arm.
- **Mortal** (Chakrabarti-Kumar-Radlinski-Upfal 2009): arms have a finite lifetime; cohort-based ε-greedy.
- **Federated** (Shi-Shen 2021): K agents, each runs UCB1 locally; periodic sync averages `(n_i, x̄_i)` across agents.
- **Constrained / Safe** (Sani-Lazaric-Munos 2012; Amani-Alizadeh-Thrampoulidis 2019): each arm has a (mean, cost-mean); never pull an arm whose UCB-cost exceeds budget B (use `optim.SimplexMethod` for budget-allocation).
- **Risk-aware / CVaR-UCB / Mean-variance** (Sani-Lazaric-Munos 2012): replace mean by `μ̂_i − ρ σ̂_i` (mean-variance) or empirical CVaR_α(rewards_i).

---

## 3. Cross-package edges (~5 keystones)

1. **`linalg.RankOneUpdate`** — Sherman-Morrison `(A + xx^T)^{-1} = A^{-1} − (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)`. Currently absent. Co-ship as 96-linalg-missing PR or here as a 30-LOC private helper. **Gates B11-B14.**
2. **`prob.GaussianProcessRegression`** — 169-S15 keystone. **Gates B17.**
3. **`prob.KLDivergence` (Bernoulli special case)** — already in `prob/info.go`. **Used by B3.**
4. **`optim.BisectionMethod` / `NewtonRaphson`** — already in `optim/rootfind.go`. **Used by B3, B4.**
5. **`graph.QuadTree` / `Dijkstra`** — `Dijkstra` already in `graph/`. QuadTree absent. **Gates B22, B24-path.**

---

## 4. Composition story

**PR-1 (one-day, ~620 LOC):** B1 + B2 + B3 + B4 + B5 + B6 (the four-Pinsker-bound stochastic family + Gittins index) — lands the `StochasticBandit` interface keystone, saturates **R-PINSKER-DUAL 4/4** + **R-CONJUGATE-FAMILY 2/2**, gates everything else.

**PR-2 (one-week, ~1,050 LOC):** B11 + B12 + B13 + B14 + B17 + B25 (LinUCB-disjoint + LinUCB-hybrid + LinTS + OFUL + GP-UCB + RUCB) — lands the contextual + GP + dueling clusters that are the actual industrial-recommender / hyperparameter-search / RLHF surface, depends on linalg `RankOneUpdate` + 169-S15 GP regression.

**PR-3 (two-week, ~1,200 LOC):** B7 + B8 + B9 + B10 + B18 + B19 + B20 + B21 + B26 + B27 + B28 — adversarial + BAI + non-stationary + restless + corner cluster.

**PR-4 (research-grade, ~530 LOC):** B15 + B16 + B22 + B23 + B24 — SquareCB + NeuralUCB + Lipschitz + HOO + Combinatorial.

---

## 5. Saturation pin candidates

| Pin | Cardinality | Constituents | Cross-validates |
|---|---|---|---|
| **R-PINSKER-DUAL** | 4/4 | UCB1, UCB-V, KL-UCB, BayesUCB on Bernoulli benchmark | regret rate matches Lai-Robbins lower bound up to constant; KL-UCB tightest (constant 1) |
| **R-FULL-VS-BANDIT-FEEDBACK** | 2/2 | Hedge × EXP3 on adversarial bilinear | identical update; estimator-only difference; `√K` cost gap |
| **R-CONJUGATE-FAMILY** | 2/2 | BayesUCB × Gaussian Thompson | Beta-Bernoulli vs Normal-Inverse-Gamma; identical posterior-quantile dynamic |
| **R-SELF-NORMALISED-MARTINGALE** | 1/1 | OFUL | de la Peña-Klass-Lai bound + LinUCB sufficient; LinUCB-without-OFUL bound is loose |
| **R-PREFERENCE-FEEDBACK** | 2/2 | RUCB × DTS | Condorcet-winner-elicit; identical regret rate |
| **R-BOB-BEST-OF-BOTH-WORLDS** | 1/1 | Tsallis-INF | matches stochastic AND adversarial bound simultaneously without regime-knowledge |
| **R-LOWER-BOUND-MATCHING-BAI** | 1/1 | Track-and-Stop | LP-allocation matches Garivier-Kaufmann lower bound asymptotically |

---

## 6. Naming-collision check

`gametheory/bandit.go` keeps its three existing functions verbatim. New file `gametheory/bandit_state.go` for B1-B6 + adversarial. New file `gametheory/contextual_bandit.go` for B11-B17. New file `gametheory/bai.go` for B18-B21. New file `gametheory/structured_bandit.go` for B22-B25. New file `gametheory/restless_bandit.go` for B26-B28. **Zero collisions with existing exports.**

---

## 7. Research-recency note

All cited references are pre-2026. The 2024-2026 frontier — **constrained-RLHF bandits** (Liu-Li-Pacchiano-Dann 2024), **federated dueling bandits** (Mao-Nguyen-Liu 2025), **transformer-as-bandit-in-context-learner** (Lin-Bai-Mei 2024), **risk-aware GP-UCB with martingale concentration** (Tu-Roberts 2025), **best-arm identification with arbitrary side information** (Atsidakou-Garivier-Kaufmann 2026) — would be a separate PR-5 (~600 LOC) tracking 2024+ NeurIPS/ICML lines. Not Tier-1; flagged here for completeness.

---

## 8. Final accounting

- **B1-B28: 28 primitives, ~3,400 LOC of pure connective tissue.**
- **One genuinely new abstraction:** the `StochasticBandit` interface (~50 LOC), co-shipped with 174-G4 `OnlineLearner` and 220-F1 `FiniteSumLoss` keystones.
- **Five cross-package edges:** linalg rank-1 update, prob GP regression (169-S15), prob Bernoulli-KL (already there), optim Bisection/Newton (already there), graph QuadTree (absent).
- **Cheapest one-day standalone:** PR-1 B1-B6 (~620 LOC).
- **Highest-leverage one-week:** PR-2 B11+B13+B14+B17+B25 (~1,050 LOC) covers contextual + GP + dueling.
- **Saturation pin shopping list:** R-PINSKER-DUAL 4/4, R-CONJUGATE-FAMILY 2/2, R-PREFERENCE-FEEDBACK 2/2, R-BOB 1/1, R-LOWER-BOUND-MATCHING-BAI 1/1, R-FULL-VS-BANDIT-FEEDBACK 2/2, R-SELF-NORMALISED-MARTINGALE 1/1.

The repo currently exposes **3 bandit functions out of a canonical surface of ~31** — **<10% coverage** of the bandit canon. After PR-1+PR-2+PR-3 the coverage rises to **~85%** (B1-B14 + B17-B21 + B25-B28), with PR-4 (B15/B16/B22-B24) being the research-grade tail. Bandits are the **highest-density-of-citations** corner of cutting-edge math reality is missing — every recommender, every hyperparameter-tuner, every clinical-trial designer, every RLHF preference-collector, every A/B-tester, every adaptive-experimentation system imports exactly this surface. The three-function `bandit.go` is a token presence; the cluster A-F enumeration above is the actual library.

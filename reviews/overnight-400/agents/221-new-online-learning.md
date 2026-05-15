# 221 | new-online-learning

**Topic:** Online learning — regret minimisation, Online Gradient Descent (Zinkevich 2003), Mirror Descent / Online Mirror Descent (Beck-Teboulle 2003 / Bubeck-Cesa-Bianchi 2012), Follow-The-Regularized-Leader (Shalev-Shwartz 2007 / McMahan 2011), Hedge / Multiplicative-Weights (Freund-Schapire 1997), adversarial bandits (EXP3 — Auer-Cesa-Bianchi-Freund-Schapire 2002), stochastic bandits (UCB — Auer-Cesa-Bianchi-Fischer 2002), contextual bandits (LinUCB — Li-Chu-Langford-Schapire 2010), Online Newton Step (Hazan-Agarwal-Kale 2007), strongly-convex / exp-concave logarithmic-regret regimes, switching-experts (Herbster-Warmuth 1998), strongly-adaptive regret (Daniely-Gonen-Shalev-Shwartz 2015), Cover's Universal Portfolio (Cover 1991), Online Matrix Prediction (Tsuda-Rätsch-Warmuth 2005), smoothed-online-learning, differentially-private online learning, online-learning-meets-game-theory cross-link. **Block:** C (cutting-edge-math, what reality is missing). **Date:** 2026-05-08.
**Scope:** the **online-convex-optimisation + experts + bandits + adaptive-regret-against-changing-comparators** axis — distinct from 174 (G4/G5/G6/G7/G9/G13-G18 partial-overlap on the equilibrium-computation slice) and 220 (F3/F4 stochastic *batch* SGD + Tier-6 schedules cross-link). 221 owns the **adversarial-loss / regret-bound / dynamic-regret / bandit-feedback** triangle and the four classes of regret bound (vanilla `O(√T)`, strongly-convex `O(log T)`, exp-concave `O(log T)`, adaptive `O(√(T·m))` for `m` shifts).

## Two-line summary

`reality/` ships ZERO online-convex-optimisation surface and ZERO regret bookkeeping — the only online-style primitives in the entire repo are the three iid-stochastic-bandit functions at `gametheory/bandit.go:37/88/195` (`UCB1`, `ThompsonSampling`, `EpsilonGreedy`) which are state-LESS (caller passes counts/rewards arrays in, gets one pull index out — no cumulative-regret, no adversarial-feedback, no streaming `Update`/`Action` interface), plus the private 100k-step fictitious-play loop at `gametheory/nash.go:170-209` (named `Minimax`, no public `OnlineLearner`); 174-G4 named the missing `OnlineLearner` interface as the architectural keystone for that synergy and 220-Tier-6 names OGD/FTRL as ~180 LOC composing `optim/proximal.ProxBox`/`ProxSimplex`/`ProxL2Ball` — **221 names the remaining 26 primitives O1-O26 totalling ~3,800 LOC of pure connective tissue covering the four-bound-class regret canon (O(√T) vanilla / O(log T) strongly-convex / O(log T) exp-concave / O(√(T·m)) shifting-comparator) plus the three feedback regimes (full-information experts / stochastic-bandit / adversarial-bandit / contextual-bandit) plus the meta-learners (switching experts, strongly adaptive, smoothed online learning) plus the matrix / portfolio / DP corners** — every primitive is a 30-200 LOC composition of the existing `optim/proximal/` projector library + 174-G4 `OnlineLearner` interface + 220-Tier-6 schedule wrappers + this review's lone new abstraction the **`AdaptiveRegretMinimizer` two-tier interface (~50 LOC)** that decorates any base `OnlineLearner` with a sleeping-experts wrapper. Cheapest one-day standalone is **PR-1 OnlineLearner+Hedge+OGD+OMD substrate (~480 LOC, co-shipped with 174-G4/G5/G13/G14)** which lands the **first regret-bookkeeping anything** in `reality/`. Highest-leverage one-week unlock is **O14 Online Newton Step + O17 Strongly-Adaptive Hedge + O18 LinUCB + O20 Universal Portfolio (~880 LOC)** because they collectively saturate the **R-REGRET-RATE-FAMILY 4/4** pin (vanilla O(√T) × strong-convex O(log T) × exp-concave O(log T) × m-switching O(√(T·m)) all match their respective lower bounds within constants on a shared bilinear-payoff benchmark). Architectural keystone is the **`OnlineLearner` interface (40 LOC, = 174-G4)** + `AdaptiveRegretMinimizer` decorator (50 LOC, new) — co-shipped with 174's PR-2 the same patch.

---

## 0. State of play (verified file-walk)

### `gametheory/` online surface = THREE state-less bandit FUNCTIONS

Verified by direct read of `bandit.go`:

- `UCB1(counts, rewards, totalPulls) → int` (`bandit.go:37-66`) — function takes the consumer's state arrays and returns one arm index. **No cumulative state, no regret, no Update method.**
- `ThompsonSampling(successes, failures, rng) → int` (`bandit.go:88-108`) — same shape. Internal Marsaglia-Tsang gamma + Box-Muller normal.
- `EpsilonGreedy(rewards, counts, epsilon, rng) → int` (`bandit.go:195-221`) — same shape.
- `Minimax(A, maxIter=100000)` at `nash.go:170-209` — private fictitious-play loop with no Brown-1951 convergence test, no average-iterate emission, no public `OnlineLearner`.

Verified ABSENT (repo-wide grep on every package):

- **No regret-anything:** zero matches for `regret`, `Regret`, `cumulativeRegret`, `dynamicRegret`, `staticRegret`, `externalRegret`, `internalRegret`, `swapRegret`, `comparatorRegret`, `R_T`.
- **No OCO primitives:** zero matches for `OnlineGradient`, `OGD`, `OnlineLearner`, `MirrorDescent`, `OMD`, `EntropicMD`, `Bregman`, `FollowTheRegularizedLeader`, `FTRL`, `FollowTheLeader`, `FTL`, `OnlineNewton`, `ONS`.
- **No experts framework:** zero matches for `Hedge`, `MultiplicativeWeights`, `MWU`, `WeightedMajority`, `expertAdvice`, `ExpertAdvice`, `predictionWithExpertAdvice`, `sleepingExpert`, `treeExpert`, `switchingExpert`.
- **No adversarial bandit:** zero matches for `EXP3`, `EXP3.P`, `EXP4`, `EXP3.IX`, `Tsallis`, `TsallisINF`, `INF`.
- **No contextual bandit:** zero matches for `LinUCB`, `linearBandit`, `LinTS`, `OFUL`, `Thompson.contextual`, `contextualBandit`, `disjointLinUCB`, `hybrid.linucb`.
- **No best-arm identification:** zero matches for `BestArmIdentification`, `BAI`, `successive.elimination`, `LUCB`, `racing`, `pure.exploration`, `lil.UCB`, `lilUCB`.
- **No portfolio:** zero matches for `UniversalPortfolio`, `Cover.portfolio`, `EG.portfolio`, `OnlineNewtonPortfolio`, `BCRP`, `BAH`.
- **No matrix online:** zero matches for `MatrixHedge`, `MatrixMWU`, `OnlineMatrix`, `Tsuda.Ratsch.Warmuth`, `vonNeumann.divergence`, `MatrixMultiplicativeWeights`, `MMW`.
- **No adaptive regret:** zero matches for `AdaptiveRegret`, `StronglyAdaptive`, `Hazan.Seshadhri`, `SAOL`, `Daniely.Gonen.ShalevShwartz`, `FLH`, `coin.betting`, `parameterFree`.
- **No smoothed / DP online:** zero matches for `SmoothedOnline`, `dispersion`, `diff.privacy`, `differentialPrivacy`, `treeAggregation`, `BLR`, `binaryMechanism`, `private.OGD`.
- **No Lipschitz / kernel bandit:** zero matches for `LipschitzBandit`, `zoomingTree`, `HOO`, `gaussianProcess.bandit`, `GPUCB`, `kernelUCB`.

**Three RNG-touching points in entire repo, named at 195-N3 / 220-keystone:** `optim/genetic.go:58-65` (Box-Muller), `optim/metaheuristic.go:38-93` (`SimulatedAnnealing` Boltzmann acceptance), `gametheory/bandit.go:113-176` (Marsaglia-Tsang gamma + Box-Muller + Beta-via-gamma — the internal sampler hidden inside `ThompsonSampling`).

### Cross-coupling: zero (verified)

```
$ grep -r "github.com/davly/reality/optim" gametheory/ ; echo "---"
$ grep -r "github.com/davly/reality/proximal" gametheory/
---
(no matches in either direction)
```

Same vacuum 174 and 220 named — **221 inherits 174-G4 OnlineLearner keystone + 195-N3 / 220 unified RNG keystone** and adds the `AdaptiveRegretMinimizer` decorator on top.

---

## 1. The conceptual unlock — **regret IS the OCO contract**, not a metric

OCO (Zinkevich 2003) reframes optimisation as a **game**: at each round `t = 1, 2, …, T`,

1. Learner chooses `x_t ∈ K` (compact convex action set).
2. Adversary reveals convex loss `ℓ_t : K → ℝ`.
3. Learner suffers `ℓ_t(x_t)`.

The **regret** against any fixed comparator `u ∈ K` is

```
R_T(u) = Σ_{t=1}^T ℓ_t(x_t) − Σ_{t=1}^T ℓ_t(u),    R_T = max_{u∈K} R_T(u)
```

A learner is "no-regret" when `R_T / T → 0` as `T → ∞`. Four lower-bound classes, each tight to a constant:

| Loss class | Rate | Algorithm |
|---|---|---|
| Convex Lipschitz | `O(√T)` | OGD (Zinkevich 2003), OMD (Beck-Teboulle 2003), FTRL (Shalev-Shwartz 2007) |
| Strongly convex | `O(log T)` | OGD-with-`η_t = 1/(μt)` (Hazan-Agarwal-Kale 2007), ONS |
| Exp-concave | `O((d/α) log T)` | Online Newton Step (Hazan-Agarwal-Kale 2007) |
| Convex + `m`-shifting | `O(√(T·m))` | Strongly-Adaptive Hedge (Hazan-Seshadhri 2009; Daniely-Gonen-Shalev-Shwartz 2015) |

The genius of the framework: **average-iterate OGD on the bilinear payoff `ℓ_t(x) = a_t^T x` recovers Hedge** (entropic-mirror-map = OMD-on-simplex, 174-G14); **OGD on `ℓ_t(x) = ½‖x − y_t‖²` recovers an online Kalman filter**; **OGD against an adversarial regression-loss recovers stochastic gradient descent** (220-F3 with `B = 1`); **OGD where the comparator is a moving target recovers tracking / dynamic-regret bounds**. One framework subsumes all of streaming-supervised-learning, all of online-game-playing, and all of dynamic-tracking. The repo's three Tier-6 220 primitives (OGD, FTRL, schedules) are its **front door**; 221's 26 primitives expand the door into a cathedral.

---

## 2. Twenty-six synergy primitives (O1-O26, ~3,800 LOC pure glue)

Numbered ascending by composition-depth. Each lists (capability, composition of existing primitives, LOC).

### Tier 0 — substrate (ships first, gates everything else; co-ships with 174-G4, 220-keystone, 195-N3)

**O1 OnlineLearner interface** [~40 LOC, **= 174-G4, ship once**]
```go
type OnlineLearner interface {
    Action() int                  // discrete arm pick (or call Distribution for mixed)
    Distribution(out []float64)   // mixed strategy / continuous action
    Update(payoff []float64)      // observed full-info payoff vector for each arm
    Regret() float64              // cumulative external-regret upper bound
}
```
Continuous-action variant `OnlineConvexLearner`:
```go
type OnlineConvexLearner interface {
    Play(x []float64)             // current action x_t ∈ K ⊂ R^d
    Update(grad []float64)        // observed sub-gradient g_t = ∇ℓ_t(x_t) (or proxy)
    Regret() float64              // cumulative external-regret upper bound
}
```
Two interfaces because discrete-experts (Hedge, EXP3) and continuous-action (OGD, OMD, ONS) have fundamentally different state shapes. Place at `gametheory/online.go` (174's recommendation).

**O2 RegretAccumulator helper** [~80 LOC, new]
Tracks `Σ ℓ_t(x_t) − min_u Σ ℓ_t(u)` numerically-stably (Kahan-Neumaier compensated summation, since cumulative regret is the *difference of two large nearly-equal sums* — classic catastrophic-cancellation hazard). Exposes `Add(loss_played, loss_comparator_min)`, `Cumulative() float64`, `RatioOverT() float64`. **Catches a hazard 174 does not name:** at `T = 10⁶` and unit-loss range, naïve `float64` summation has relative error `~10⁻¹⁰`, swamped by a true `O(√T) = O(1000)` regret signal.

### Tier 1 — full-information OCO (composes Tier 0 + `optim/proximal/`)

**O3 OnlineGradientDescent** (Zinkevich 2003 *ICML*) [~80 LOC, **= 174-G13 = 220-Tier-6, ship once**]
`x_{t+1} = Π_K(x_t − η_t g_t)` for `g_t ∈ ∂ℓ_t(x_t)`; `η_t = D / (G √t)`. Composes `optim/proximal.ProxBox`/`ProxSimplex`/`ProxL2Ball`/`ProxNonNeg` as the projection oracle. Implements `OnlineConvexLearner`. **Regret `R_T ≤ DG√T`** where `D = diam(K)`, `G = sup ‖∂ℓ_t‖`. Document the Hazan-Kale 2014 *JMLR* tight constant `R_T ≤ (3/2)DG√T` for the optimal step-size.

**O4 OnlineMirrorDescent** (Beck-Teboulle 2003 *Op. Res. Letters* 31:167; Bubeck-Cesa-Bianchi 2012 §4) [~120 LOC, **= 174-G14, ship once**]
Generalised `Π_K` via Bregman divergence `D_R(x,y) = R(x) − R(y) − ⟨∇R(y), x − y⟩` for a strictly-convex mirror map `R`. Update: `∇R(x_{t+1}) = ∇R(x_t) − η g_t` then project by `D_R`. Closed-form for two cases:
- `R(x) = ½‖x‖²` → OMD = OGD (O3).
- `R(x) = Σ x_i log x_i − x_i` (negative entropy) on simplex → `x_{t+1,i} ∝ x_{t,i} exp(−η g_{t,i})` = **Hedge** (O8). The single composition that unifies five algorithms in 174-PR-2's R-OMD-UNIFIES-FIVE-LEARNERS pin.

**O5 LazyOMD / DualAveraging** (Nesterov 2009 *Math. Programming* 120; Xiao 2010 *JMLR* 11) [~100 LOC]
Dual-averaging variant: maintain `z_t = Σ_{s≤t} g_s` in the dual, return `x_{t+1} = argmin_{x∈K} ⟨z_t, x⟩ + (1/η_t) R(x)`. Equivalent to OMD up to a step-size redefinition; better stability under non-smooth / sparse gradients (O11 RDA below is the regularised variant). Composes `optim/proximal.ProxL1` for sparse case.

**O6 FollowTheRegularizedLeader** (Shalev-Shwartz 2007 *PhD thesis*; McMahan 2011 *AISTATS*) [~100 LOC, **= 174-G15 = 220-Tier-6, ship once**]
`x_{t+1} = argmin_{x∈K} Σ_{s≤t} ⟨g_s, x⟩ + R_t(x)/η`. With `R(x) = ½‖x‖²` → lazy OGD; with `R(x) = entropy` → lazy Hedge. **FTRL = Lazy-OMD when proximal step is exact** (Hazan 2016 *Foundations and Trends in Optim* 2(3-4):157-325 §5.5). Composes `optim.LBFGS` for the per-iter argmin when no closed-form. Cross-link to `optim/proximal.ProxL1` for **FTRL-Proximal** (McMahan 2013 *KDD* — Google's billion-feature sparse-logistic-regression deployment).

**O7 FollowTheLeader (FTL) + counterexamples** [~50 LOC]
`x_{t+1} = argmin Σ_{s≤t} ℓ_s(x)`. **The naïve baseline that motivates regularisation.** Document the Cesa-Bianchi-Lugosi 2006 §3.2 worst-case construction where FTL has linear regret `Ω(T)` on adversarial linear losses (alternating `+1/-1` sequence). Pin: FTL on `K = [-1, 1]` with `ℓ_t(x) = sign(t)·x` has `R_T ≥ T` while O3 OGD has `R_T = O(√T)`. **First negative-result pin in the repo** (R-FTL-LINEAR-REGRET 1/1).

### Tier 2 — experts / multiplicative weights (composes Tier 0 + `prob.SoftmaxStable`)

**O8 Hedge / Multiplicative Weights Update** (Freund-Schapire 1997 *J. Comp. Sys. Sci.* 55(1):119; Littlestone-Warmuth 1994 *Inf. Comp.* 108(2):212; Arora-Hazan-Kale 2012 *Theory Computing* 8) [~80 LOC, **= 174-G5, ship once**]
`p_{t+1,i} ∝ p_{t,i} · (1 − η ℓ_{t,i})` (Littlestone-Warmuth) or `p_{t+1,i} ∝ p_{t,i} · exp(−η ℓ_{t,i})` (Freund-Schapire — the modern form). External regret `R_T ≤ √(2T log K)` with optimal `η = √(2 log K / T)`. **Identical update rule appears in over 100 ML / TCS algorithms** (boosting, set-cover, max-flow, online task allocation, MWU-as-LP-solver). The Arora-Hazan-Kale 2012 survey explicitly identifies this as *the* primal-dual primitive of online TCS.

**O9 ProdNormalHedge / NormalHedge** (Chaudhuri-Freund-Hsu 2009 *NIPS*; Cesa-Bianchi-Mansour-Stoltz 2007 *Mach. Learn.* 66) [~120 LOC]
Parameter-free Hedge variant — no `η` to tune. Achieves regret to **top-ε quantile of experts** at rate `O(√(T·log(1/ε)))` for any user-specified `ε ∈ (0, 1]`. Crown-jewel for adversarial-experts when no horizon `T` is known and Hedge's `η = √(log K/T)` cannot be set. Reference Koolen-van-Erven 2015 *COLT* "Second-order Quantile Methods" for the modern variant.

**O10 RegretMatching / RegretMatching+** (Hart-Mas-Colell 2000 *Econometrica* 68(5):1127) [~70 LOC, **= 174-G6, ship once**]
`p_{t+1,i} ∝ max(R_t,i⁺, 0)` with `R_t,i = Σ_{s≤t}(r_s,i − r_s,a_s)`. RM+ truncates negative cumulative regrets per round. Internal-regret variant → **coarse correlated equilibrium** (cross-link 174-G19 CCE-LP). The CFR family (174-G11) is regret-matching at every information set.

**O11 ExponentiatedGradientRDA / RegularizedDualAveraging** (Kivinen-Warmuth 1997 *Inf. Comp.* 132(1):1; Xiao 2010 *JMLR* 11:2543) [~100 LOC]
Exponentiated-gradient is OMD with negative-entropy on the positive orthant (no-simplex-constraint variant). Regularised-Dual-Averaging Xiao 2010 is the L1-regulariser FTRL variant — **the algorithm running ad-click prediction in production** at scale (Google paper); `optim/proximal.ProxL1` is the per-iter projection.

### Tier 3 — bandit feedback (composes Tier 2 + RNG keystone)

**O12 EXP3 / EXP3.P / EXP3.IX** (Auer-Cesa-Bianchi-Freund-Schapire 2002 *SIAM J. Comput.* 32(1):48; Neu 2015 *NIPS* "Explore no More") [~140 LOC, **= 174-G9, ship once for EXP3; +60 LOC for EXP3.IX implicit-exploration variant**]
Adversarial-bandit Hedge: only the played arm's payoff is observed. Importance-weight estimator `ĝ_t,i = g_t,i · 𝟙[a_t = i] / p_t,i`. Regret `O(√(T K log K))` with EXP3, `O(√(T K log K))` w.h.p. with EXP3.P (Auer 2003 high-probability variant). EXP3.IX (Neu 2015) has `O(√(T K))` with implicit exploration; cleaner constants.

**O13 UCB1 / UCB-V / UCB-Tuned / KL-UCB** (Auer-Cesa-Bianchi-Fischer 2002 *Mach. Learn.* 47:235; Audibert-Munos-Szepesvari 2009 *Theor. Comput. Sci.* 410:1876; Garivier-Cappé 2011 *COLT*) [~120 LOC, **builds-on-existing `bandit.go:UCB1`**]
Stochastic-bandit upper-confidence-bound family. **`gametheory/bandit.go:UCB1` is `UCB1` only — three variants missing:**
- UCB-V: replaces `√(2 log T / n_i)` exploration with empirical-variance-aware `√(2 σ̂² log T / n_i) + 3 log T / n_i`. Tighter on low-variance arms. Audibert-Munos-Szepesvari 2009.
- UCB-Tuned: Auer-Cesa-Bianchi-Fischer 2002 §4.2 — empirically dominant variant; `min(¼, σ̂² + √(2 log T / n_i))` exploration.
- KL-UCB: Garivier-Cappé 2011 — uses the KL-divergence quantile rather than the Hoeffding bound; **matches the Lai-Robbins 1985 lower bound asymptotically**. Composes `prob`'s KL functions (cross-link 170 info-prob).

State-FUL wrapper that satisfies `OnlineLearner` (current `bandit.go:UCB1` is state-LESS — caller manages `counts`/`rewards`).

### Tier 4 — stochastic-and-adaptive regret (composes Tier 1 + Tier 3)

**O14 OnlineNewtonStep** (Hazan-Agarwal-Kale 2007 *Mach. Learn.* 69:169) [~180 LOC]
For **exp-concave** losses (`exp(−α ℓ_t)` concave): `x_{t+1} = Π^A_t_K (x_t − (1/β) A_t^{-1} g_t)` where `A_t = Σ_{s≤t} g_s g_s^T + ε I` is the **empirical Fisher / outer-product accumulator**. Regret `O((1/α + GD) · d log T)`. **Logarithmic regret** — unique among Tier 1-3. Composes `linalg.Inverse` (or rank-1 Sherman-Morrison `A_t^{-1} = A_{t-1}^{-1} − A_{t-1}^{-1} g_t g_t^T A_{t-1}^{-1} / (1 + g_t^T A_{t-1}^{-1} g_t)` for `O(d²)` per-step instead of `O(d³)`). Cross-link 220-F8 Adam: **Adam is heuristic-O(d) per-coordinate ONS** with diagonal `A_t` instead of full matrix.

**O15 ScaleFreeFTRL / AdaFTRL / AdaGrad-Online** (Orabona-Pál 2018 *Theor. Comput. Sci.* 716:50; Duchi-Hazan-Singer 2011 *JMLR* 12:2121) [~120 LOC]
**Scale-free**: regret bound holds simultaneously across all gradient scales without tuning. AdaGrad-online uses `η_t,i = D / √(Σ_{s≤t} g_s,i²)` per-coordinate. **Adagrad's online-to-batch reduction** (Cesa-Bianchi-Conconi-Gentile 2004) is what makes 220-F10 AdaGrad work — the regret bound *is* the convergence rate.

**O16 AdaptiveOnlineLearning / FreezingThaw / FollowTheLeadingHistory** (Hazan-Seshadhri 2009 *ICML* "Efficient Learning Algorithms for Changing Environments") [~180 LOC]
**Strongly-adaptive regret** = regret on every contiguous interval `[r, s] ⊆ [1, T]`: `R_{[r,s]}(u) = Σ_{t=r}^s ℓ_t(x_t) − Σ_{t=r}^s ℓ_t(u) ≤ O(√(s-r))`. Hazan-Seshadhri's FLH (Follow-the-Leading-History) algorithm achieves this for any `[r, s]` simultaneously. Geometric-grid tree-of-experts wrapper composes `OnlineLearner`s as black boxes — every Tier-1/2/3 primitive lifts to its strongly-adaptive variant by passing it through O16.

**O17 SAOL / CoinBetting** (Daniely-Gonen-Shalev-Shwartz 2015 *ICML* "Strongly Adaptive Online Learning"; Orabona-Pál 2016 *NIPS* "Coin Betting and Parameter-Free Online Learning") [~150 LOC]
SAOL: improvement over O16 — strongly-adaptive regret `O(√(s-r) log s)` (vs. `log s` factor in O16). Coin-betting: parameter-free OCO via the Kelly-criterion / log-wealth duality (cross-link `gametheory.Kelly`!) — **regret `O(D √T) `without tuning**, where `D` is diameter of the optimal comparator. **First gametheory.Kelly cross-consumer in the repo.**

### Tier 5 — switching / tracking / dynamic regret (composes Tier 4)

**O18 SwitchingExperts / FixedShare** (Herbster-Warmuth 1998 *Mach. Learn.* 32(2):151) [~110 LOC]
Hedge variant that allows the comparator to switch up to `m` times across `[1, T]`. Update mixes a uniform-noise-floor `α/K` into the post-Hedge distribution before normalisation: `p_{t+1,i} ∝ (1−α) · p_t,i exp(−η ℓ_t,i) + α/K`. Regret to `m`-shifting comparator `O(√(T(m log K + m log T)))`. The original "switching-experts" algorithm.

**O19 TreeExperts / SpecialistsHedge** (Freund-Schapire-Singer-Warmuth 1997 *STOC*; Cohen-Mansour 2017 *NeurIPS*) [~140 LOC]
Hedge over a tree of experts where each leaf is a "specialist" active only on a subset of rounds (the **sleeping-experts** model). Generalises switching-experts: switch-points are tree-structured. **Cross-link to 174-G11 CFR** (which is RM at every info-set node — same tree-hedge primitive, different feedback model).

**O20 LearningRestarts / DoublingTrick / SquintHedge** (Cesa-Bianchi 2012 *Bernoulli*; Koolen-van-Erven 2015 *COLT*) [~120 LOC]
Doubling-trick: run a Tier-1 learner with horizon `T_k = 2^k`; restart at every doubling. Achieves anytime regret `O(√T)` from a fixed-horizon learner. Squint (Koolen-van-Erven 2015) is the optimal-second-order parameter-free variant; hits the `O(√(T·V_T))` second-order bound where `V_T` is the loss-variance — strictly stronger than vanilla O8 Hedge.

### Tier 6 — contextual bandits (composes Tier 1 + Tier 3)

**O21 LinUCB (disjoint + hybrid)** (Li-Chu-Langford-Schapire 2010 *WWW* "A Contextual-Bandit Approach to Personalized News Article Recommendation") [~200 LOC]
Linear contextual bandit: arm `i` reward `r_t,i = x_t^T θ_i* + noise`. Maintain `A_i = Σ x_s x_s^T + I`, `b_i = Σ r_s x_s`; play `argmax_i x_t^T (A_i^{-1} b_i) + α √(x_t^T A_i^{-1} x_t)`. Regret `O(d √T)`. Disjoint variant has separate `θ_i*` per arm; hybrid shares features. **The contextual-bandit primitive deployed at Yahoo News, MSN, Microsoft Ads.** Composes `linalg`'s rank-1 inverse update (Sherman-Morrison) for `O(d²)` per round.

**O22 LinThompsonSampling** (Agrawal-Goyal 2013 *ICML* "Thompson Sampling for Contextual Bandits with Linear Payoffs") [~140 LOC]
Bayesian variant of O21: maintain Gaussian posterior over `θ_i*`, sample `θ̃_i ~ N(μ_i, v² A_i^{-1})`, play `argmax x_t^T θ̃_i`. Frequentist regret `O(d^{3/2} √T)`. Composes the Marsaglia-Tsang Gaussian sampler already inside `gametheory/bandit.go`. **First Bayesian-online-learning primitive in repo.**

**O23 OFUL / KernelUCB / GP-UCB** (Abbasi-Yadkori-Pál-Szepesvari 2011 *NIPS*; Valko-Korda-Munos-Cristianini 2013 *ICML*; Srinivas-Krause-Kakade-Seeger 2010 *ICML*) [~220 LOC]
OFUL: tighter LinUCB constants via self-normalised concentration. KernelUCB: lift to RKHS via kernel `K(x, x')`. GP-UCB: Gaussian-process variant; regret `O(√(T·γ_T))` where `γ_T` is the maximum information gain over `T` rounds. **Lipschitz-bandit and continuous-arm bandits emerge here** — Lipschitz arms become a kernel-bandit instance with the Matérn kernel.

### Tier 7 — pure exploration / best-arm identification (orthogonal axis)

**O24 SuccessiveElimination / LUCB / LIL'UCB / Track-and-Stop** (Even-Dar-Mannor-Mansour 2006 *JMLR* 7:1079; Kalyanakrishnan-Tewari-Auer-Stone 2012 *ICML*; Jamieson-Malloy-Nowak-Bubeck 2014 *COLT*; Garivier-Kaufmann 2016 *COLT* "Optimal Best-Arm Identification with Fixed Confidence") [~250 LOC]
**Pure-exploration regime**: minimise sample complexity for `(ε, δ)`-best-arm-identification, NOT regret. SE: drop arms whose UCB falls below another arm's LCB. LUCB: simultaneously play the leader and the highest-UCB challenger. lil'UCB: law-of-iterated-logarithm tight constants. Track-and-Stop (Garivier-Kaufmann 2016): asymptotically optimal sample complexity matching the Chernoff lower bound. **The minimum-regret-vs-best-arm-identification tradeoff** is a fundamental open question — Bubeck-Munos-Stoltz 2009 *COLT* prove the two objectives are at odds (a regret-minimizer cannot also be sample-optimal for BAI).

### Tier 8 — portfolio / matrix / smoothed / DP (orthogonal axes)

**O25 UniversalPortfolio / EG-Portfolio / OnlineNewtonPortfolio** (Cover 1991 *Math. Finance* 1(1):1; Helmbold-Schapire-Singer-Warmuth 1998 *Math. Finance* 8(4):325; Hazan-Agarwal-Kale-Singer-Bartlett 2007 *NIPS*) [~200 LOC]
**Cover's universal portfolio**: for `n`-asset market with daily-return vectors `r_t`, the algorithm `b_{t+1} = ∫ b · S_t(b) db / ∫ S_t(b) db` (where `S_t(b) = Π_{s≤t} ⟨b, r_s⟩` is wealth) achieves wealth within `O(1/T^{n-1})` of the **best constant-rebalanced portfolio** in hindsight. Cover's original 1991 algorithm is `O(T^n)` — exponential in `n`. EG (Helmbold et al. 1998) is `O(nT)` with `O(log n / √T)` regret. ONS-Portfolio (Hazan et al. 2007) achieves `O(d log T)` regret via O14 Online Newton Step. **First financial / log-wealth primitive in repo** (cross-link `gametheory.Kelly`!).

**O26 MatrixHedge / MatrixMultiplicativeWeights** (Tsuda-Rätsch-Warmuth 2005 *JMLR* 6:995 "Matrix Exponentiated Gradient Updates"; Arora-Kale 2007 *STOC* "Combinatorial, Primal-Dual Approach to Semidefinite Programs") [~180 LOC]
Generalises Hedge from simplex to **density-matrix simplex** `{X ⪰ 0, tr(X) = 1}`. Update: `X_{t+1} ∝ exp(log X_t − η L_t)` where the matrix-exp-and-log are operator functions (composable with `linalg.MatrixExp`/`MatrixLog` if those land — flagged as 188-T-prob×linalg gap). Bregman divergence is **von Neumann entropy** `D(X, Y) = tr(X log X − X log Y − X + Y)`. Regret `O(√(T log d))`. **First quantum-information primitive in repo**; powers SDP solvers, online metric learning, and quantum learning theory. Cross-link 188 prob-linalg, 203 tensor-networks, 217 free-prob.

**Deferred to v2 (named but not budgeted):**
- **Smoothed Online Learning** (Haghtalab-Roughgarden-Shetty 2022 *J. ACM*) — adversary draws `ℓ_t` from a `σ`-smooth distribution over a finite hypothesis class; regret `O(√(T log(1/σ)))`. Defer; specialised to learning theory.
- **Differentially-Private Online Learning** (Jain-Kothari-Thakurta 2012 *NIPS* "Differentially Private Online Learning"; Smith-Thakurta 2013 *NIPS*) — privatised OGD via tree-aggregation noise mechanism. Defer until 175 zkmark-crypto / 200 zkmark-info land the Laplace-mechanism / Gaussian-mechanism / tree-aggregation primitives.
- **Meta-Learning of Online Learning Rates** (Awerbuch-Kleinberg 2008 *J. Comp. Sys. Sci.*; Cesa-Bianchi-Mansour-Stoltz 2007) — second-order parameter-tuning Hedge; defer.
- **Multi-Agent Online Learning** (Daskalakis-Fishelson-Golowich 2021 *NIPS*; Anagnostides-Daskalakis-Farina-Sandholm 2022 *NeurIPS*) — when all players run `O(log T)`-regret algorithms, joint dynamics converge to coarse correlated equilibrium at rate `O(1/T)` (vs. `O(1/√T)` for vanilla OGD/Hedge). Cross-link 174-G19. Defer.
- **Online learning with delayed feedback** (Joulani-Gyorgy-Szepesvari 2013 *ICML*) — `O(√(T(d+1)))` regret with `d`-step delay. Defer.

---

## 3. Composition graph (DAG)

```
O1 OnlineLearner / OnlineConvexLearner ifc (= 174-G4, 40 LOC)  ┐
O2 RegretAccumulator (Kahan-Neumaier compensated, 80 LOC)      ├── architectural keystones
RNG ifc (= 195-N3 = 220-keystone, 80 LOC)                       ┘

Tier 1: full-info OCO
O3 OGD (= 174-G13 = 220-T6) — composes optim/proximal.Prox*
O4 OMD (= 174-G14)
 ├── O5 LazyOMD / DualAveraging
 ├── O6 FTRL (= 174-G15 = 220-T6) — composes optim.LBFGS
 └── O7 FTL + counterexample pin

Tier 2: experts
O8 Hedge (= 174-G5)              ┐
 ├── O9 NormalHedge (param-free)  │── experts cluster
 ├── O10 RegretMatching (= 174-G6)│
 └── O11 EG-RDA (sparse FTRL)     ┘

Tier 3: bandit
O12 EXP3 (= 174-G9) ── composes O8 + RNG (importance weighting)
O13 UCB family (UCB1+UCB-V+UCB-Tuned+KL-UCB) — extends bandit.go:UCB1

Tier 4: adaptive
O14 OnlineNewtonStep (composes linalg.Inverse + Sherman-Morrison)
O15 AdaGradOnline / ScaleFreeFTRL (= 220-F10 online-form)
O16 FLH adaptive-regret tree
O17 SAOL / CoinBetting (composes gametheory.Kelly)

Tier 5: tracking
O18 SwitchingExperts / FixedShare (composes O8 Hedge)
O19 TreeExperts / Specialists (composes O8 + tree)
O20 DoublingTrick / Squint (composes O8 + restart)

Tier 6: contextual bandits
O21 LinUCB (composes linalg.Inverse + Sherman-Morrison)
O22 LinThompson (composes O21 + Marsaglia-Tsang sampler from bandit.go)
O23 OFUL / KernelUCB / GP-UCB (composes O21 + kernel evaluator)

Tier 7: pure exploration (orthogonal)
O24 SuccessiveElim / LUCB / lilUCB / Track-and-Stop (composes prob.KL for KL-LCB)

Tier 8: orthogonal corners
O25 UniversalPortfolio / EG-Portfolio / ONS-Portfolio (composes O14 + gametheory.Kelly)
O26 MatrixHedge / MMW (composes linalg.MatrixExp/MatrixLog if shipped, else 188-gap)

Cross-cutting wrappers (from 220-T6 schedules)
 - cosine / step / inverse-time → η_t schedules feed every Tier-1/2/3
```

---

## 4. Saturation pins this review unlocks

Per the audio-onset 6a55bb4 / Clayton-autodiff 365368a / NGramDice 85a80db idiom:

- **R-REGRET-RATE-FAMILY 4/4 (O3, O8, O14, O17):** four regret-rate classes match their respective lower bounds within constants on a bilinear-payoff `ℓ_t(x) = ⟨a_t, x⟩` benchmark on the simplex with `T = 10^4, K = 10`. Vanilla O3 OGD: `R_T / √T → const`. Hedge O8: `R_T / √(T log K) → const`. ONS O14 on quadratic-loss: `R_T / log T → const`. SAOL O17 on m-shifting comparator with m=5: `R_T / √(Tm log K) → const`. **Four orthogonal regret rates simultaneously certified — saturates the OCO lower-bound hierarchy.**
- **R-OMD-UNIFIES-FIVE-LEARNERS 5/5 (O4 OMD subsumes):** entropic OMD with vanishing-η lazy projection ≡ O8 Hedge with cumulative-payoff. O8 Hedge with β→∞ ≡ FollowTheLeader specialisation (174-G7 fictitious play). O4 OMD with squared-Euclidean mirror ≡ O3 OGD. O4 OMD with negative-entropy mirror on positive orthant ≡ O11 EG. **Five algorithms collapse to one composition tree — the cleanest didactic exhibit in the entire repo's online surface.** This is 174-PR-2's pin, but 221 makes it 5/5 by adding O7 FTL counterexample as the negative pole.
- **R-LAST-ITERATE-VS-AVERAGE-ITERATE 2/2 (174-PR-3 lift):** vanilla OGD on bilinear Matching-Pennies cycles in last-iterate (Bailey-Piliouras 2018) but average-iterate converges at `O(1/√T)`; Optimistic-OGD (174-G16, 220-out-of-scope) achieves last-iterate `O(1/T)`. **Already named in 174-PR-3** — 221 cross-references.
- **R-BANDIT-REGRET-LOWER-BOUNDS 3/3 (O12, O13, O21):** EXP3 hits `O(√(TK log K))` adversarial-bandit lower bound (Auer-Cesa-Bianchi-Freund-Schapire 2002 §6); KL-UCB hits Lai-Robbins 1985 stochastic-bandit lower bound `Σ Δ_i / KL(p_i ‖ p*)`; LinUCB hits `Ω(d√T)` linear-bandit lower bound (Dani-Hayes-Kakade 2008 *COLT*). **Three feedback regimes simultaneously matching their respective tight lower bounds.**
- **R-FTL-LINEAR-REGRET 1/1 (O7 negative pin):** FTL on `K = [-1, 1]` with adversarial losses `ℓ_t(x) = sign(t) · x` has `R_T = T` (linear in T) while OGD on the same sequence has `R_T = O(√T)`. **First negative-result pin in the repo** — every review names what works; this names what *fails* and *why* regularisation is mandatory.
- **R-COVER-UNIVERSAL-PORTFOLIO 1/1 (O25):** EG-Portfolio on a 5-asset 252-day synthetic stationary market achieves wealth within `O(log n / √T) = O(0.05)` of the best-constant-rebalanced portfolio in hindsight per Helmbold et al. 1998 Theorem 1. **First financial-time-series pin in repo.**
- **R-MATRIX-HEDGE-VS-VECTOR-HEDGE 1/1 (O26 generalisation):** O8 Hedge on `R^d` simplex ≡ O26 MatrixHedge on the diagonal `d × d` density-matrix subspace. Generalisation pin (one specialisation reduces to another in the limit).

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| O1 | OnlineLearner / OnlineConvexLearner ifc (= 174-G4) | 40 | 0 | — |
| O2 | RegretAccumulator (Kahan-Neumaier compensated) | 80 | 0 | — |
| O3 | OGD (= 174-G13 = 220-T6) | 80 | 1 | O1, optim/proximal |
| O4 | OMD (= 174-G14) | 120 | 1 | O1, prob.Softmax |
| O5 | LazyOMD / DualAveraging | 100 | 1 | O4 |
| O6 | FTRL (= 174-G15 = 220-T6) | 100 | 1 | O1, optim.LBFGS |
| O7 | FTL + counterexample pin | 50 | 1 | — |
| O8 | Hedge / MWU (= 174-G5) | 80 | 2 | O1, prob.Softmax |
| O9 | NormalHedge / parameter-free | 120 | 2 | O8 |
| O10 | RegretMatching/+ (= 174-G6) | 70 | 2 | O1 |
| O11 | EG-RDA (sparse FTRL) | 100 | 2 | O5, optim/proximal.ProxL1 |
| O12 | EXP3 / EXP3.P / EXP3.IX (= 174-G9) | 200 | 3 | O8, RNG |
| O13 | UCB family (UCB1/UCB-V/UCB-Tuned/KL-UCB) | 120 | 3 | bandit.go:UCB1, prob.KL |
| O14 | Online Newton Step | 180 | 4 | O1, linalg.Inverse + Sherman-Morrison |
| O15 | ScaleFreeFTRL / AdaGrad-Online (= 220-F10 online-form) | 120 | 4 | O6 |
| O16 | FLH adaptive-regret tree | 180 | 4 | O8 (any inner) |
| O17 | SAOL / Coin-Betting | 150 | 4 | O8, gametheory.Kelly |
| O18 | SwitchingExperts / FixedShare | 110 | 5 | O8 |
| O19 | TreeExperts / Specialists | 140 | 5 | O8, tree DS |
| O20 | DoublingTrick / Squint | 120 | 5 | O8 |
| O21 | LinUCB (disjoint + hybrid) | 200 | 6 | linalg.Inverse + Sherman-Morrison |
| O22 | LinThompsonSampling | 140 | 6 | O21, RNG (Marsaglia-Tsang) |
| O23 | OFUL / KernelUCB / GP-UCB | 220 | 6 | O21, kernel eval |
| O24 | SuccessiveElim / LUCB / lilUCB / Track-and-Stop | 250 | 7 | prob.KL |
| O25 | UniversalPortfolio / EG-Portfolio / ONS-Portfolio | 200 | 8 | O14, gametheory.Kelly |
| O26 | MatrixHedge / MMW | 180 | 8 | linalg.MatrixExp/Log (188-gap) |
| **Σ** | | **3,750** | | |

(Excluding the deduplicated overlap with 174 + 220: **net new LOC for 221 ≈ 2,950** since O1, O3, O4, O6, O8, O10, O12 would be co-shipped under either review heading — 221 owns the **regret-bookkeeping + adaptive-regret + bandit-feedback + contextual-bandit + portfolio + matrix-online + pure-exploration** axes; 174 owns the **equilibrium-computation slice** of the same OCO core; 220 owns the **batch-finite-sum-SGD** axis.)

Pure-glue ratio: ~80 % composition over `optim/proximal.Prox*` projectors + 174-G4 `OnlineLearner` interface + 220-T6 schedule wrappers + `bandit.go` Marsaglia-Tsang sampler + `linalg.Inverse` rank-1 Sherman-Morrison + `prob.KL`/`SoftmaxStable`. ~20 % genuinely-new math (O14 ONS Sherman-Morrison rank-1 update at ~80 LOC; O16 FLH tree-of-experts at ~120 LOC; O17 coin-betting Kelly-duality at ~80 LOC; O25 universal-portfolio integration / EG-portfolio at ~140 LOC; O26 matrix-exp Bregman at ~120 LOC).

---

## 6. Recommended PR sequence

**PR-1: substrate (O1 OnlineLearner + O2 RegretAccumulator + O3 OGD + O4 OMD + O8 Hedge) — ~480 LOC source, ~280 LOC tests, single day**
**Co-shipped with 174-PR-1/PR-2 the same patch.** Lands the **first regret-bookkeeping anything** in `reality/`. The five primitives form the closed core that every subsequent PR composes. Saturates R-OMD-UNIFIES-FIVE-LEARNERS 5/5 immediately (O4 OMD = O3 OGD = O8 Hedge under three distinct mirror-map specialisations). Lifts the private FP loop at `nash.go:170-209` to a public `OnlineLearner` (174-G7 lift).

**PR-2: classical OCO + experts (O5 LazyOMD + O6 FTRL + O7 FTL-counterexample + O9 NormalHedge + O10 RegretMatching + O11 EG-RDA) — ~520 LOC source, ~280 LOC tests, one day**
Completes the experts canon. **Saturates R-FTL-LINEAR-REGRET 1/1** (negative-result pin on FTL). Composes `optim/proximal.ProxL1` for FTRL-Proximal sparse logistic regression (McMahan 2013 KDD).

**PR-3: bandit feedback (O12 EXP3 family + O13 UCB family with UCB-V/UCB-Tuned/KL-UCB) — ~320 LOC source, ~180 LOC tests, half-day**
Lifts the existing `bandit.go:UCB1` to state-FUL `OnlineLearner`-conformant struct + adds three UCB variants. Saturates **R-BANDIT-REGRET-LOWER-BOUNDS 3/3** for stochastic + adversarial regimes (LinUCB linear-bandit pin lands in PR-6).

**PR-4: adaptive regret + scale-free (O14 Online Newton Step + O15 AdaGrad-Online + O16 FLH + O17 SAOL/CoinBetting) — ~630 LOC source, ~340 LOC tests, two days**
**The adaptive-regret cluster.** Saturates **R-REGRET-RATE-FAMILY 4/4**: ONS hits `O(d log T)` for exp-concave; SAOL hits `O(√(Tm log K))` for m-shifting; vanilla O3 hits `O(√T)`; combined with O8 Hedge's `O(√(T log K))` from PR-1, four regret rates simultaneously match their lower bounds. **Highest-leverage PR for cross-validation pins.** O17 is the first `gametheory.Kelly` cross-consumer in the repo.

**PR-5: tracking / switching (O18 FixedShare + O19 TreeExperts + O20 DoublingTrick + Squint) — ~370 LOC source, ~200 LOC tests, one day**
Switching-comparator family. Composes O8 Hedge as black-box; geometric-grid restart for anytime regret.

**PR-6: contextual bandits (O21 LinUCB + O22 LinThompson + O23 OFUL/KernelUCB/GP-UCB) — ~560 LOC source, ~300 LOC tests, two days**
**The personalised-recommendation primitive.** O21 LinUCB is the Yahoo / MSN / Microsoft-Ads workhorse. O22 LinThompson reuses the Marsaglia-Tsang sampler already inside `bandit.go`. **Saturates R-BANDIT-REGRET-LOWER-BOUNDS 3/3** (third pin: LinUCB matches `Ω(d√T)`). Cross-link 188 prob-linalg for rank-1 Sherman-Morrison.

**PR-7: pure exploration (O24 SuccessiveElim + LUCB + lilUCB + Track-and-Stop) — ~250 LOC source, ~140 LOC tests, one day**
Best-arm-identification regime (NOT regret minimisation). Composes `prob.KL` for KL-LCB confidence radii. Cross-link to A/B-testing, drug-trial, and clinical-decision consumers.

**PR-8: portfolio + matrix online (O25 UniversalPortfolio/EG-Portfolio/ONS-Portfolio + O26 MatrixHedge) — ~380 LOC source, ~220 LOC tests, one and a half days**
**The two orthogonal corners.** O25 is the first financial / log-wealth primitive in repo (cross-link `gametheory.Kelly`); saturates **R-COVER-UNIVERSAL-PORTFOLIO 1/1**. O26 is the first density-matrix / quantum-information primitive in repo; cross-link 188 prob-linalg, 203 tensor-networks, 217 free-prob.

**Total: 8 PRs, ~3,510 LOC source + ~1,940 LOC tests across ~10 engineer-days.** PR-1 is co-shipped with 174-PR-1/PR-2 the same patch (zero marginal LOC for the substrate). PR-4 (adaptive-regret cluster) is the single highest-utility PR for downstream consumers since it lands the `O(log T)` regime that batch SGD (220) cannot reach. PR-6 (LinUCB) is the single highest-utility PR for production consumers (recommendation systems). PR-8 (Universal Portfolio + MatrixHedge) is the crown-jewel composition.

---

## 7. Cycle-hazard analysis

Proposed import directions:

```
gametheory/   ──→  optim/proximal/       (O3, O4, O5, O11, O18 — projectors)
gametheory/   ──→  optim/                (O6 FTRL via optim.LBFGS, O14 ONS via linalg)
gametheory/   ──→  prob/                 (O8 SoftmaxStable, O13 KL-UCB, O24 KL-confidence)
gametheory/   ──→  linalg/               (O14 ONS, O21 LinUCB, O26 MatrixExp/Log)
optim/        ──→  gametheory/           (NONE — opposite direction reserved)
```

**No new cross-package edges beyond what 174 already proposed** + a new `gametheory/ → linalg/` edge (for O14, O21, O26). 174's PR-1 already establishes `gametheory/ → optim/` and `gametheory/ → optim/proximal/`; 221 reuses both. The `linalg/` edge is new but trivial (only `linalg.Inverse` + Sherman-Morrison rank-1 utility).

**Crucially: 221 does NOT add any reverse edges** — `optim/`, `prob/`, `linalg/` never import `gametheory/`. The DAG remains cycle-free. Verified by enumeration.

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **O2 RegretAccumulator:** cumulative regret is the *difference of two large nearly-equal sums* (played-loss-sum minus comparator-loss-sum). At `T = 10⁶` and unit-loss range, naïve `float64` summation has relative error `~10⁻¹⁰`, swamped by a true `O(√T) = O(1000)` regret signal. **Mandate Kahan-Neumaier compensated summation; document the precision floor**. Existing `prob/` does not have a Kahan utility — 220-keystone names the absence; 221 adds the streaming-regret variant.
- **O3 OGD step-size η_t = D/(G√t):** requires known diameter `D = diam(K)` and Lipschitz constant `G = sup ‖∂ℓ_t‖`. If `G` unknown: use AdaGrad-Online (O15) which is scale-free. Document the trade-off.
- **O4 OMD Bregman convexity:** mirror map `R` must be `α`-strongly-convex over `K` w.r.t. some norm. Failure mode: non-strongly-convex `R` causes update non-uniqueness. Document the canonical pairs `R(x) = ½‖x‖²` (1-strongly-convex w.r.t. Euclidean norm) and `R(x) = Σ x_i log x_i` (1-strongly-convex w.r.t. L1 norm on the simplex by Pinsker's inequality).
- **O8 Hedge step-size η:** theoretical optimum `η = √(2 log K / T)` requires known horizon `T`. For anytime variant use doubling trick (O20) or `η_t = √(log K / t)`. **Same hazard 174 names** — document and default to anytime.
- **O10 RegretMatching ties:** when all positive-cumulative-regrets are zero (start of run, or after RM+ truncation), uniform play is the canonical fallback (Hart-Mas-Colell 2000 fn. 4). **174 names this precision pin** — 221 cross-references.
- **O12 EXP3 importance-weight variance blowup:** `r̂_t,i = r_t,i / p_t,i · 𝟙[a_t = i]` has variance `~1/min(p)`. Min-prob mixing `p_t ← (1−γ) p_t + γ/K` is the standard fix (Auer 2002 EXP3 with `γ = √(K log K / T)`). EXP3.IX (Neu 2015) eliminates the explicit mixing via implicit exploration in the loss estimator — cleaner.
- **O13 KL-UCB:** the `KL_inf(p_i, μ*) = sup_q { KL(p_i ‖ q) : q.mean ≤ μ* }` requires Newton's method on the KL functional (no closed-form). Composes existing `optim.NewtonRaphson`. Document the convergence floor `1e-12` and the `100`-iteration cap.
- **O14 ONS:** `A_t = Σ g_s g_s^T + ε I` requires `ε > 0` for invertibility — Hazan 2007 takes `ε = 1/(β² D²)` where `β = α/2 ∧ G/D`. Sherman-Morrison rank-1 update is `O(d²)` per step; vanilla matrix inverse is `O(d³)` — document the 100x speed difference at `d = 100`.
- **O17 Coin-Betting / Kelly duality:** Kelly's `b* = E[x]/Var[x]` (the leveraged-Kelly fraction) is the **optimal regret-free play in continuous-action coin-betting**; the connection to log-wealth duality is non-obvious. Reference Orabona-Pál 2016 *NIPS* §3 for the formal duality theorem.
- **O18 SwitchingExperts mixing-rate α:** Herbster-Warmuth 1998 §5 set `α = (m + 1) / T` for known `m` (number of switches) and known `T`. Without `m`: use prior `α_0 = 1/T` and document the `O(√(T m log T))` regret instead of `O(√(T m log K))`. **m-aware vs m-blind regret rates differ by a `log(T/K)` factor**.
- **O20 Doubling-trick discontinuity:** at every doubling boundary `T = 2^k → 2^(k+1)`, the per-instance learner restarts and discards all state — the per-instance regret is `O(√(2^k))` and the global regret is `Σ_k √(2^k) = O(√T)`, BUT the algorithm is **piecewise non-anytime** (regret bound holds only at doubling boundaries, not in between). For continuous-time anytime use Squint (Koolen-van-Erven 2015) instead.
- **O21 LinUCB confidence radius α:** Li-Chu-Langford-Schapire 2010 use `α = 1 + √(ln(2/δ)/2)` heuristically; theoretical regret bounds (Abbasi-Yadkori-Pál-Szepesvari 2011 OFUL) need `α = √(d ln((1+T·L²)/δ)) + ‖θ*‖` for high-probability `O(d√T)` — much larger than the LCS heuristic. Document the **theory-vs-practice gap** and recommend OFUL constants for theoretical guarantees, LCS heuristic for empirical performance.
- **O22 LinThompson posterior variance:** Bayesian linear regression with Gaussian likelihood gives `θ̃ ~ N(A^{-1} b, σ² A^{-1})`. The frequentist regret bound (Agrawal-Goyal 2013) requires sampling from `N(A^{-1} b, v² A^{-1})` with **inflated** variance `v² = R √(d log(t/δ))` where `R` is the noise sub-Gaussian parameter. The naïve posterior under-explores. Document the inflation factor.
- **O23 GP-UCB information-gain γ_T:** for Matérn kernel with smoothness `ν > d/2`, `γ_T = O(T^{(d/(2ν+d)) log T})` — sub-linear only when `ν > d/2`. RBF kernel: `γ_T = O((log T)^{d+1})`. Document the kernel-dependent regret rate.
- **O24 Track-and-Stop:** Garivier-Kaufmann 2016 require **known sub-exponential parameter `b`** for the confidence radius. Without `b`: empirical-Bernstein + iterated logarithm gives `(1 + o(1))` of the optimal sample complexity but loses the `(1 + o(1))` to a `log log T` factor.
- **O25 UniversalPortfolio Cover-1991 cost:** original Cover 1991 algorithm is `O(T^n)` per round (Dirichlet integral over portfolio simplex). EG-Portfolio (Helmbold et al. 1998) is `O(nT)` total but with `O(log n / √T)` regret instead of Cover's `O(n log T / T)`. Document the **exponential-time / polynomial-time trade-off**.
- **O26 MatrixHedge update:** `X_{t+1} ∝ exp(log X_t − η L_t)` requires symmetric matrix-exp / matrix-log via eigendecomposition (`O(d³)` per step). Cross-link 188 prob-linalg's `linalg.MatrixExp` proposal (currently absent).

---

## 9. Distinct from prior agents (provenance)

- **011-015 autodiff** — orthogonal axis; OCO is first-order with hand-derivable gradients (every primitive O3-O26 has closed-form `∇ℓ_t`).
- **071-075 gametheory isolation** — names "fictitious play embedded in `Minimax` should be public no-regret iterator" (071-§F1, 074-T2.5); 221 cross-references via O1 OnlineLearner = 174-G4. **221 is the OCO-canonical view; 174 is the equilibrium-computation view; both consume the same `OnlineLearner` interface.**
- **101-105 optim isolation** — names Adam/Lion/SVRG missing (102-T1.6-T1.8); 220 fills these. **221 is strictly orthogonal**: every O-primitive consumes the existing optim+proximal+linalg surface; no new optim primitives are required (only the `optim/proximal.ProxL1`/`ProxBox`/`ProxSimplex`/`ProxL2Ball` projectors which already ship).
- **151-153, 161-168 prior synergies** — orthogonal axes (signal-prob, prob-infogeo, crypto-prob, control-prob, graph-prob, orbital-optim, sequence-prob, acoustics-signal, audio-signal, physics-autodiff). O14 ONS shares the empirical-Fisher-matrix theme with 153 prob-infogeo's natural-gradient; cross-link only.
- **163 synergy-optim-autodiff** — names forward-mode duals + HVP; **221 is strictly orthogonal**: O-primitives are first-order. If 163 lands, O6 FTRL gains the option of reverse-mode-AD-supplied gradient closure to the inner-argmin LBFGS — decoration, not necessity.
- **169 synergy-prob-optim (deterministic-fit half)** — orthogonal: 169 owns MLE/MAP/EM/VI/BO; 221 owns adversarial-regret. Cross-link only via 169-S15 BBVI which uses score-function gradient — same gradient-closure pattern as 221's O3 OGD on stochastic loss but different regime (offline VI vs. online OCO).
- **170 synergy-info-prob, 171 graph-topology, 172 changepoint-timeseries, 173 queue-prob** — orthogonal.
- **174 synergy-gametheory-optim (PARTIAL OVERLAP, MAJOR)** — 174 owns the **equilibrium-computation slice** (G1-G3 LP, G10-G12 CFR, G19-G20 CCE/Stackelberg). **221 owns the regret-bookkeeping + adaptive + bandit + contextual + portfolio + matrix-online slices**. Overlap: 174-G4 (OnlineLearner = O1), G5 (Hedge = O8), G6 (RegretMatching = O10), G7 (FP — orthogonal to 221, ships in 174 only), G9 (EXP3 = O12), G13 (OGD = O3), G14 (OMD = O4), G15 (FTRL = O6). For the overlap set: **co-ship under both review headings, ship code once**. Net new in 221 vs 174: O2 (numerically-stable regret accumulator), O5 (lazy OMD), O7 (FTL counterexample pin), O9 (NormalHedge / parameter-free), O11 (EG-RDA), O13 (UCB family extending bandit.go), O14 (Online Newton Step), O15 (AdaGrad-Online), O16 (FLH adaptive), O17 (SAOL / Coin-Betting), O18 (SwitchingExperts), O19 (TreeExperts), O20 (DoublingTrick / Squint), O21 (LinUCB), O22 (LinThompson), O23 (OFUL / KernelUCB / GP-UCB), O24 (BAI family), O25 (UniversalPortfolio), O26 (MatrixHedge) — i.e., **18 of 26 primitives are unique to this slot**; 8 of 26 are shared with 174 and ship once.
- **175-219 prior synergies + Block-C reviews** — orthogonal except where noted (O14 ONS shares Sherman-Morrison theme with 213 NLA-randomized; O21 LinUCB shares rank-1 inverse update with 215 compressed-sensing iterative-reweighting; O25 UniversalPortfolio has financial axis like nothing else in repo; O26 MatrixHedge shares density-matrix simplex with 188 prob-linalg, 203 tensor-networks, 217 free-prob).
- **220 synergy-stochastic-opt (PARTIAL OVERLAP, MINOR)** — 220 owns **finite-sum batch / mini-batch / momentum / adaptive-batch / variance-reduced / second-order / federated**; 221 owns **adversarial-loss / regret-bookkeeping / dynamic-comparator / bandit-feedback**. Overlap: 220-Tier-6 names OGD/FTRL/schedules at ~430 LOC; **221 absorbs that exact line item as O3/O6 plus the schedule wrappers** — co-ship. Net new in 221 vs 220: 23 of 26 primitives. The conceptual unification: **online-to-batch reduction (Cesa-Bianchi-Conconi-Gentile 2004 *IEEE Trans Inf Theory* 50:2050) shows that any `R_T = O(T^α)` no-regret OCO algorithm gives a `O(T^(α-1))` stochastic-batch convergence rate** — i.e., O8 Hedge with `R_T = O(√(T log K))` automatically gives a `1/√T` convergence rate for stochastic optimisation on the simplex; 220-F8 Adam's empirical success is heuristic-O14 ONS via the online-to-batch reduction. **Pin this identity** as the conceptual bridge between 220 (batch) and 221 (online).

**18 of 26 primitives unique to this slot.** 221 is the **regret-bound-canonical** slot in the 400-sequence — every primitive named here is what 2026 ML / TCS literature calls by these specific names (Hedge, MWU, OGD, OMD, FTRL, EXP3, UCB-V, KL-UCB, ONS, AdaGrad-Online, LinUCB, LinThompson, OFUL, GP-UCB, FixedShare, Squint, FLH, SAOL, Coin-Betting, NormalHedge, RM/RM+, Universal Portfolio, EG-Portfolio, MatrixHedge, MMW, Successive Elimination, LUCB, lilUCB, Track-and-Stop) rather than the equilibrium-computation aliases 174 gives them or the batch-finite-sum aliases 220 gives them.

---

## 10. Bottom line

`reality/` ships ZERO online-convex-optimisation surface and ZERO regret bookkeeping despite `gametheory/bandit.go` shipping three iid-stochastic-bandit utility functions (`UCB1`, `ThompsonSampling`, `EpsilonGreedy`) with a state-LESS calling convention and despite `optim/proximal/` shipping the exact projector library OCO needs (`ProxBox`, `ProxSimplex`, `ProxL2Ball`, `ProxNonNeg`, `ProxL1`). **Twenty-six primitives O1-O26 totalling ~3,750 LOC of pure connective tissue** stand up the entire OCO + experts + bandits + adaptive-regret + contextual-bandit + portfolio + matrix-online + best-arm-identification canon on existing v0.10.0 surfaces (specifically: `optim/proximal/Prox*` projectors, `optim.LBFGS` for FTRL inner-argmin, `optim.NewtonRaphson` for KL-UCB confidence-quantile inversion, `linalg.Inverse` + Sherman-Morrison rank-1 updates for ONS / LinUCB / LinThompson, `prob.SoftmaxStable` for Hedge / OMD-entropic, `prob.KL` for KL-UCB / KL-LCB, `gametheory.bandit.go`'s Marsaglia-Tsang / Box-Muller samplers for LinThompson, `gametheory.Kelly` for SAOL coin-betting + ONS-Portfolio).

Cheapest single-day standalone is **PR-1 (O1 OnlineLearner + O2 RegretAccumulator + O3 OGD + O4 OMD + O8 Hedge = 480 LOC)** which lands the **first regret-bookkeeping anything** in `reality/` and gates O5-O26 simultaneously. **Co-shipped with 174-PR-1/PR-2 the same patch** (zero marginal LOC for the OnlineLearner interface). Single highest-utility PR for theoretical pins is **PR-4 (O14 ONS + O15 AdaGrad-Online + O16 FLH + O17 SAOL/Coin-Betting = 630 LOC)** — saturates R-REGRET-RATE-FAMILY 4/4 across all four loss-class lower bounds. Single highest-utility PR for production consumers is **PR-6 (O21 LinUCB + O22 LinThompson + O23 OFUL/KernelUCB/GP-UCB = 560 LOC)** — the personalised-recommendation primitive deployed at Yahoo / MSN / Microsoft Ads. Crown-jewel composition is **PR-8 (O25 UniversalPortfolio + O26 MatrixHedge = 380 LOC)** — first financial / log-wealth primitive AND first density-matrix / quantum-information primitive in repo, both consuming `gametheory.Kelly` + `linalg.MatrixExp`/`MatrixLog` (188-gap).

**Reality is unusually well-positioned for this synergy because (i) `gametheory/bandit.go:UCB1/ThompsonSampling/EpsilonGreedy` already ships three online-decision functions awaiting a stateful `OnlineLearner` wrapper; (ii) `gametheory/nash.go:170-209` already inlines fictitious-play awaiting public lift to `OnlineLearner` (174-G7 names this); (iii) `optim/proximal/` already ships every projector OCO needs; (iv) `gametheory/bandit.go:113-176` already ships Marsaglia-Tsang gamma + Box-Muller normal that LinThompson needs; (v) `gametheory/Kelly` already ships the log-wealth primitive that SAOL coin-betting + ONS-Portfolio compose; (vi) the consumer-side-placement convention (per agents 158-220) recommends O1-O26 in `gametheory/online*.go` (already RNG-aware, already fictitious-play-aware) rather than a new sub-package — minimum architectural perturbation, maximum OCO unlock.**

The single most important conceptual identity 221 pins: **Zinkevich 2003 OGD + Beck-Teboulle 2003 OMD + Shalev-Shwartz 2007 FTRL + Freund-Schapire 1997 Hedge + Hart-Mas-Colell 2000 RM + Auer-Cesa-Bianchi-Freund-Schapire 2002 EXP3 + Auer-Cesa-Bianchi-Fischer 2002 UCB1 + Hazan-Agarwal-Kale 2007 ONS + Cover 1991 Universal Portfolio + Tsuda-Rätsch-Warmuth 2005 MatrixHedge are ten specialisations of the same projection-onto-Bregman-ball operator** — varying only in (a) the projection-set `K` (simplex, ball, box, density-matrix simplex, portfolio simplex), (b) the Bregman divergence `D_R` (squared-Euclidean, KL, von Neumann, Itakura-Saito), (c) the feedback regime (full-info, bandit, contextual, semi-bandit), and (d) the comparator class (fixed, m-shifting, dynamic, sleeping, exp-concave). **R-OMD-UNIFIES-TEN-LEARNERS saturation lands when PR-4 ships.** Reality should pin this identity as the single canonical entry point for the entire online-learning literature.

---

*References (selected): Cesa-Bianchi-Lugosi 2006 *Prediction, Learning, and Games* (Cambridge); Hazan 2016 *Foundations and Trends in Optim* 2(3-4):157-325 *Introduction to Online Convex Optimization*; Bubeck-Cesa-Bianchi 2012 *Foundations and Trends in ML* 5(1):1-122 *Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems*; Lattimore-Szepesvari 2020 *Bandit Algorithms* (Cambridge); Shalev-Shwartz 2012 *Foundations and Trends in ML* 4(2):107-194 *Online Learning and Online Convex Optimization*; Slivkins 2019 *Foundations and Trends in ML* 12(1-2):1-286 *Introduction to Multi-Armed Bandits*; Orabona 2023 *A Modern Introduction to Online Learning* arXiv:1912.13213.*

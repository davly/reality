# 174 | synergy-gametheory-optim

**Topic:** gametheory × optim — best-response dynamics, no-regret learning, online convex optimisation, equilibrium computation.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities emerging ONLY when `gametheory/` × `optim/` (× transient `optim/proximal/`, `chaos/ode.go`) compose. Not isolation gaps (agents 071-075, 101-105 already enumerate those). Repo v0.10.0, 1965 tests passing.

## Two-line summary

`gametheory/` ships exactly two equilibrium primitives — `NashEquilibrium2x2` (closed-form 2×2) and `Minimax` (a private fictitious-play loop hard-wired at `nash.go:170-209`) — plus bandits/matching/voting/Kelly (~1100 LOC, ZERO imports of `optim/` even though the topic-prompt's "linear-programming form of zero-sum minimax" maps line-for-line onto `optim.SimplexMethod`); `optim/` ships `GradientDescent`/`LBFGS`/`SimplexMethod`/`InteriorPoint` plus `optim/proximal/` (FBS/FISTA/ADMM with eight prox ops including `ProxSimplex` and `ProxBox` — the EXACT projectors no-regret algorithms need); zero cross-edges in either direction confirmed by direct grep. **Twenty synergy primitives (G1-G20) totalling ~2,580 LOC of pure connective tissue** stand up the entire best-response/no-regret/online-convex/CFR/equilibrium-computation canon as compositions of existing optim + gametheory + (optionally) chaos primitives — only one genuinely new abstraction (the `OnlineLearner` streaming interface, ~40 LOC) is required, all twenty primitives ship today against v0.10.0.

---

## Bases — what each package exposes today

### `gametheory/` (~1100 LOC, 6 files)
- `nash.go` (~233): `NashEquilibrium2x2` (closed-form 2×2 mixed/pure NE), `Minimax` (private fictitious-play loop, `maxIter=100000`, no convergence check, no public best-response interface).
- `bandit.go` (~222): `UCB1`, `ThompsonSampling`, `EpsilonGreedy` — already-no-regret-style learners over discrete arms but not exposed as the no-regret `OnlineLearner` interface.
- `matching.go`, `voting.go`, `kelly.go`: orthogonal axes (Gale-Shapley, Banzhaf/Shapley, Kelly).

### `optim/` (~1940 LOC, 7 files + 2 sub-packages)
- `gradient.go`: `GradientDescent` (signature `func(x, g []float64)`), `LBFGS` (Armijo line search).
- `linear.go`: `SimplexMethod` (Bland's rule, standard form `min cᵀx s.t. Ax≤b, x≥0`), `InteriorPoint` (barrier-gradient).
- `rootfind.go`: scalar `BisectionMethod`, `NewtonRaphson`, `GoldenSectionSearch`.
- `metaheuristic.go`, `genetic.go`: SA, GA — gradient-free.
- `optim/proximal/`: FBS/FISTA/ADMM + `ProxSimplex` (~12 LOC), `ProxBox`, `ProxNonNeg`, `ProxL1`, `ProxL2Ball`, `ProxLinear`. The simplex projector is the **mirror-descent KL projection in disguise** for the unit simplex.
- `optim/transport/`: Sinkhorn (log-domain alternating dual potentials — same alternating-best-response skeleton as fictitious play).

### `chaos/ode.go` (transient, used by G3 only)
`RK4Step(f func(t,y,dydt), t, y, dt, out)`, `EulerStep`, `SolveODE` — exactly the signature replicator dynamics needs.

### Cross-coupling: zero today
`grep -E '"github.com/davly/reality/(optim|gametheory)"' gametheory/*.go optim/*.go` returns zero in both directions. The fictitious-play loop at `gametheory/nash.go:170-209` is private, ad-hoc, has no convergence test, and explicitly cites Brown-1951 in its docstring (`nash.go:131`) — flagged as ripe for promotion to a public no-regret interface by 071 §F1, 074-T2.5. The topic-prompt's "LP for zero-sum (use optim.SimplexMethod)" is a literally-zero-LOC composition that has not been written yet.

---

## Twenty synergy primitives

Every primitive is a **pure composition** of existing surface — no new mathematics is required beyond a 40-LOC `OnlineLearner` streaming interface (G7). Primitives are grouped by capability cluster.

### Cluster A — best-response & equilibrium computation via LP (G1-G3)

**G1. ZeroSumViaSimplex** (~80 LOC). Solve `max_p min_q pᵀAq` for an `m×n` zero-sum matrix `A` via the standard LP reformulation: `max v` s.t. `Aᵀp ≥ v·1`, `1ᵀp = 1`, `p ≥ 0`. Build LP from `A`, call `optim.SimplexMethod`, extract `(p, q)` from primal+dual basis. Replaces the 100k-iteration fictitious-play in `Minimax` for accuracy-critical callers; converges to machine precision in finite pivots vs. 1/√T fictitious play. Reference: Karlin (1959) *Mathematical Methods in Games*. Numerically validates against existing `Minimax` to ~1e-6 (R-MUTUAL-CROSS-VALIDATION 3/3 candidate: simplex-LP × FP × G2-Lemke-Howson on zero-sum subset).

**G2. LemkeHowsonBimatrix** (~280 LOC). Lexicographic pivoting on the labelled `(A,B)` linear complementarity problem. References: Lemke-Howson (1964); von Stengel ch. 3 in *Handbook of Game Theory* vol. 3. Dropped-label re-seeding enumerates multiple NE. Composes `optim.SimplexMethod`'s Bland-pivot tableau machinery as a private subroutine (extract `pivot()` from `linear.go` to internal helper). 072 T1-3 already names this at ~300 LOC; this review adds the G1 cross-validation pin.

**G3. ReplicatorDynamics** (~140 LOC). `ẋᵢ = xᵢ(fᵢ(x) − φ(x))` where `fᵢ(x) = (Ax)ᵢ`, `φ(x) = xᵀAx`. Compose `chaos.RK4Step` with simplex projection `proximal.ProxSimplex` after each step (handles numerical drift off the simplex). ESS test on Hessian. Symmetric-matrix-game NE = stable rest points. Reference: Taylor-Jonker (1978); Hofbauer-Sigmund (1998) *Evolutionary Games and Population Dynamics*. 072 T2-1 names this at ~120 LOC chaos-coupled — this review adds the simplex-projection step the consumer can otherwise drift off.

### Cluster B — no-regret learning interface (G4-G9)

**G4. OnlineLearner interface** (~40 LOC, the only genuinely new abstraction). Streaming online-learner contract:

```go
type OnlineLearner interface {
    Action() int                  // current play (or distribution)
    Distribution(out []float64)   // current mixed strategy
    Update(payoff []float64)      // observed payoff vector for each action
    Regret() float64              // cumulative external regret bound
}
```

Mirror existing `bandit.go` style (single-method functions taking `interface{ Float64() float64 }` rng) — but state is now object-resident because no-regret algorithms need cumulative quantities. Place in `gametheory/online.go`.

**G5. Hedge / MultiplicativeWeightsUpdate** (Freund-Schapire 1997, Arora-Hazan-Kale 2012) (~80 LOC). `pᵢ ∝ exp(η·Σₜ rᵢₜ)`. Implements `OnlineLearner`. Convergence: external regret `O(√(T log K))`. Two-player zero-sum self-play with both sides running Hedge → average iterate is ε-Nash at rate `O(1/√T)` (Freund-Schapire theorem). Reference Cesa-Bianchi-Lugosi (2006) *Prediction, Learning, and Games* ch. 2.

**G6. RegretMatching / RegretMatching+** (Hart-Mas-Colell 2000) (~70 LOC). `pᵢ ∝ max(R⁺ᵢ, 0)`, where `Rᵢ = Σₜ(rᵢₜ − rₐₜ)`. RM+ truncates negative cumulative regrets. Implements `OnlineLearner`. Internal-regret variant → coarse correlated equilibrium. Reference: Hart-Mas-Colell (2000) *Econometrica* 68(5):1127-1150.

**G7. FictitiousPlay (public)** (Brown 1951; Robinson 1951) (~60 LOC). Lift the private loop at `nash.go:170-209` to a public `OnlineLearner`-conformant struct. Robinson convergence theorem proves average-iterate convergence for zero-sum (the existing private loop is already this — but it lacks the streaming interface, regret bookkeeping, or stop criterion). Composes `Action()` = best-response argmax-row over cumulative-payoff (pure). 074-T2.5 names this.

**G8. SmoothFictitiousPlay** (Fudenberg-Levine 1995) (~50 LOC). G7 with logit best-response: `pᵢ ∝ exp(βr̄ᵢ)`. β→∞ recovers G7; β=1 recovers G5/Hedge with cumulative payoff. Single primitive bridges fictitious play and exponential weights — direct line of code from `r̄` array to `prob.SoftmaxStable` (already in prob/). Reference: Fudenberg-Levine (1998) *The Theory of Learning in Games* ch. 4.

**G9. ExpWeightsBandit** (Auer-Cesa-Bianchi-Freund-Schapire 2002 EXP3) (~100 LOC). G5 adapted to bandit feedback (only the played arm's payoff is observed, importance-weight to estimate full vector). Implements `OnlineLearner`. Cross-link to `bandit.go`: where UCB1 is for stochastic-iid arms, EXP3 is for adversarial-arm bandits — both consume identical state. Reference: Auer et al. (2002) *SIAM J. Comput.* 32(1):48-77.

### Cluster C — counterfactual-regret minimisation for extensive-form games (G10-G12)

**G10. ExtensiveFormGame interface** (~80 LOC). Minimal extensive-form game tree representation: information sets, actions, terminal payoffs. Just enough to make G11/G12 substrate-agnostic. Mirrors OpenSpiel's `Game/State` reduction. Place at `gametheory/extensive.go`. (Out-of-scope for v0.10.0 ship: a real game library — but the substrate the CFR family consumes lives here.)

**G11. CFR / CFR+ / LinearCFR / DiscountedCFR** (Zinkevich-Johanson-Bowling-Piccione 2007; Tammelin-Burch-Johanson-Bowling 2014; Brown-Sandholm 2019) (~340 LOC). Per-information-set local regret-matching (G6) on the EF tree → ε-Nash for zero-sum. CFR+ uses RM+ (G6 variant). LinearCFR/DiscountedCFR weight iterates by `t` or `t^α`. **Composes G6 RegretMatching at every information set** → the entire CFR family is one local online-learner per node. Reference: Zinkevich et al. (2007) *NIPS*; Brown-Sandholm (2017) *Science* 359 (DeepStack/Libratus poker); Brown-Sandholm (2019) *AAAI*.

**G12. MCCFROutcomeSampling / ExternalSampling** (Lanctot-Waugh-Zinkevich-Bowling 2009) (~180 LOC). Sample-based variant of G11; touches one trajectory per iteration. Composes `gametheory.bandit.go`'s `interface{ Float64() float64 }` RNG contract. Reference: Lanctot et al. (2009) *NIPS*.

### Cluster D — online convex optimisation (G13-G17)

**G13. OnlineGradientDescent** (Zinkevich 2003) (~60 LOC). `xₜ₊₁ = Π_K(xₜ − ηₜ∇ℓₜ(xₜ))` for compact convex `K`. Compose `optim/proximal.ProxBox` / `proximal.ProxSimplex` / `proximal.ProxL2Ball` as the projection oracle `Π_K`. Step `ηₜ = η₀/√t` → regret `O(D·G·√T)`. Implements an analogue of `OnlineLearner` over convex parameter space (action = continuous `xₜ`). Reference: Zinkevich (2003) *ICML*.

**G14. OnlineMirrorDescent / EntropicOMD** (Beck-Teboulle 2003; Bubeck-Cesa-Bianchi 2012) (~90 LOC). Generalised projection via Bregman divergence: for entropic-mirror-map on the simplex, OMD update reduces exactly to `pᵢ ∝ pᵢ exp(−η∇ℓ)` — i.e. **G5 Hedge is OMD with negative-entropy mirror map**. Cross-link makes this single composition unify five algorithms. Reference: Bubeck-Cesa-Bianchi (2012) *Foundations and Trends in ML* 5(1):1-122 ch. 4.

**G15. FollowTheRegularizedLeader** (Shalev-Shwartz 2012) (~80 LOC). `xₜ₊₁ = argmin_{x∈K} Σ_{s≤t}⟨g_s,x⟩ + R(x)/η`. With `R(x) = ½‖x‖²`: lazy-projection OGD. With `R(x) = Σxᵢlog xᵢ`: lazy Hedge. Composes `optim.LBFGS` (or `optim.GradientDescent`) for the per-iter argmin when no closed-form. Reference: Shalev-Shwartz (2012) *Foundations and Trends in ML* 4(2):107-194.

**G16. OptimisticOGD / OptimisticOMD** (Rakhlin-Sridharan 2013; Daskalakis-Ilyas-Syrgkanis-Zeng 2018) (~70 LOC). `xₜ₊₁ = Π_K(xₜ − η(2gₜ − gₜ₋₁))` — the "predict-the-next-gradient" variant. Last-iterate convergence in zero-sum games at rate `O(1/T)` vs `O(1/√T)` for vanilla OGD's average iterate. The 2018 result that solved the long-standing last-iterate gap in min-max optimisation. Reference: Daskalakis-Ilyas-Syrgkanis-Zeng (2018) *ICLR*.

**G17. ExtragradientKorpelevich / OGDA** (Korpelevich 1976; Mokhtari-Ozdaglar-Pattathil 2020) (~70 LOC). `x_{t+½} = Π(xₜ − ηF(xₜ)); xₜ₊₁ = Π(xₜ − ηF(x_{t+½}))` for variational inequality `F`. For min-max: `F = (∇_x f, −∇_y f)`. Mokhtari-Ozdaglar-Pattathil (2020) prove OGDA = optimistic-OGD = single-step extragradient under quadratic interpolation — making G16 and G17 sibling primitives. Composes `optim.GradientDescent`'s gradient closure twice per step. Reference: Korpelevich (1976) *Ekonomika i Matematicheskie Metody* 12(4); Mokhtari-Ozdaglar-Pattathil (2020) *AISTATS*.

### Cluster E — Frank-Wolfe & equilibrium-LP (G18-G20)

**G18. FrankWolfe / ConditionalGradient** (Frank-Wolfe 1956; Jaggi 2013) (~110 LOC). `sₜ = argmin_{s∈K}⟨∇f(xₜ), s⟩; xₜ₊₁ = (1−γ)xₜ + γsₜ`. Linear-oracle method for compact convex `K`. For `K = simplex`: linear oracle = pick min-coordinate basis vector (closed form, zero-LP). For `K = polytope {Ax≤b}`: linear oracle = `optim.SimplexMethod`. Sparse iterates (k-th iterate is convex combo of ≤k vertices). Reference: Jaggi (2013) *ICML* "Revisiting Frank-Wolfe". Cross-link: equilibrium-finding via FW on simplex × simplex recovers regret-matching's update.

**G19. CorrelatedEquilibriumLP** (~150 LOC). Correlated equilibrium of a finite game = LP feasibility on the joint distribution `μ ∈ Δ(S₁ × … × Sₙ)` subject to no-deviation constraints. Direct call to `optim.SimplexMethod` (or `optim.InteriorPoint`) on the LP. CCE relaxation drops internal-regret constraints and uses external. Reference: Aumann (1974) *Journal of Mathematical Economics*; Hart-Mas-Colell (2000). Composes G6 RegretMatching's CCE-converging average iterate as cross-validation source — pin: empirical CCE from G6 over 10⁵ rounds within ε=1e-3 of CCE LP solution.

**G20. StackelbergBilevelLP** (~180 LOC). For two-player Stackelberg leader-follower games with finite actions, the leader's optimal commitment-to-mixed-strategy can be computed by solving `n` LPs (one per follower pure best-response) and taking the max. Each LP composes `optim.SimplexMethod`. Reference: Conitzer-Sandholm (2006) *EC* "Computing the optimal strategy to commit to". For continuous bilevel (G20-cont): outer = `optim.LBFGS`, inner = `optim.SimplexMethod` or `optim.NewtonRaphson` for leader's best-response. Cross-link to RubberDuck's auction strategies (cited in `gametheory/nash.go:9`).

---

## Composition matrix

| # | Primitive | gametheory uses | optim uses | other | LOC |
|---|---|---|---|---|---|
| G1 | ZeroSumViaSimplex | (replaces Minimax internals) | `SimplexMethod` | — | 80 |
| G2 | LemkeHowsonBimatrix | `NashEquilibrium2x2` (cross-validate) | `SimplexMethod` pivot kernel | — | 280 |
| G3 | ReplicatorDynamics | — | `proximal.ProxSimplex` | `chaos.RK4Step` | 140 |
| G4 | OnlineLearner ifc | — | — | — | 40 |
| G5 | Hedge | `OnlineLearner` | — | `prob.SoftmaxStable` | 80 |
| G6 | RegretMatching/+ | `OnlineLearner` | — | — | 70 |
| G7 | FictitiousPlay (public) | `OnlineLearner` | — | — | 60 |
| G8 | SmoothFictitiousPlay | G5+G7 | — | `prob.SoftmaxStable` | 50 |
| G9 | EXP3 | G5, `bandit.go` rng ifc | — | — | 100 |
| G10 | ExtensiveForm ifc | — | — | — | 80 |
| G11 | CFR / CFR+ / LinCFR / DCFR | G6, G10 | — | — | 340 |
| G12 | MCCFR (sampled) | G11, `bandit.go` rng ifc | — | — | 180 |
| G13 | OnlineGradientDescent | G4 | `proximal.ProxBox`/`Simplex`/`L2Ball` | — | 60 |
| G14 | OnlineMirrorDescent | G4, G5 (entropic limit) | — | — | 90 |
| G15 | FTRL | G4 | `LBFGS` (per-iter argmin) | — | 80 |
| G16 | OptimisticOGD/OMD | G13/G14 | `proximal.Prox*` | — | 70 |
| G17 | Extragradient/OGDA | G13 | `proximal.Prox*` | — | 70 |
| G18 | FrankWolfe | — | `SimplexMethod` (linear oracle) | — | 110 |
| G19 | CorrelatedEquilibriumLP | G6 (cross-validate) | `SimplexMethod` | — | 150 |
| G20 | Stackelberg bilevel LP | — | `SimplexMethod`, `LBFGS` | — | 180 |
| **Σ** | | | | | **2,580** |

Genuinely new abstractions: **one** (G4 `OnlineLearner`, ~40 LOC). All other primitives are mechanical compositions of existing surface plus the algorithm's published update rule.

---

## Recommended PR sequence

### PR-1 — LP-zero-sum + best-response (~360 LOC, half-day)
G1 ZeroSumViaSimplex + G7 FictitiousPlay (public) + G2 LemkeHowsonBimatrix.
- Saturates **R-MUTUAL-CROSS-VALIDATION 3/3 pin** on zero-sum: `Minimax` (FP 100k iters) × `ZeroSumViaSimplex` (LP exact) × `LemkeHowsonBimatrix` (LCP exact-on-rational). Three orthogonal mathematical reductions agree to ~1e-6 on Rock-Paper-Scissors / Matching Pennies / Colonel Blotto golden vectors — mirrors commit `6a55bb4` audio-onset 3-detector and `365368a` Clayton-autodiff-vs-analytic idioms.
- First-ever cross-edge `gametheory/ → optim/`; one-way only; cycle-free verified.
- Lifts the private fictitious-play loop at `nash.go:170-209` from "embedded heuristic with no convergence test" to a public `OnlineLearner` with a documented Robinson-1951 convergence rate and an external LP-exact baseline to validate against — the same architectural lift agent 074-T2.5 names.

### PR-2 — no-regret family (~360 LOC, one day)
G4 OnlineLearner + G5 Hedge + G6 RegretMatching/+ + G8 SmoothFictitiousPlay + G14 OnlineMirrorDescent.
- Saturates **R-OMD-UNIFIES-FIVE-LEARNERS** witness pin: G14-OMD with negative-entropy mirror map, η→0 lazy-projection limit ≡ G5-Hedge-with-cumulative-payoff. G8-SmoothFictitiousPlay-β=∞ ≡ G7. G8-β=η ≡ G5. G6-RegretMatching at the no-internal-regret level → CCE. Five algorithms collapse into one composition tree → one of the cleanest didactic exhibits across the entire repo.
- Cross-link `bandit.go` UCB1 / Thompson / EpsilonGreedy and these new no-regret learners under one `OnlineLearner` interface — single API for stochastic-iid (UCB1) and adversarial (Hedge/RM/EXP3) online decisions.

### PR-3 — online convex optimisation (~370 LOC, one day)
G13 OnlineGradientDescent + G15 FTRL + G16 OptimisticOGD + G17 Extragradient + G18 FrankWolfe.
- First cross-edge `gametheory/ → optim/proximal/`. Composes the four canonical OCO updates over the existing prox-op library — `ProxSimplex` covers entropic OMD, `ProxBox` covers box-constrained OGD, `ProxL2Ball` covers ball-constrained OGD, `ProxNonNeg` covers non-negative orthant.
- Saturates **R-LAST-ITERATE-VS-AVERAGE-ITERATE 2/2 pin**: G13-OGD on bilinear Matching-Pennies has `O(1/√T)` average-iterate convergence but cycles in last-iterate (Bailey-Piliouras 2018); G16-Optimistic-OGD = G17-OGDA achieves last-iterate `O(1/T)` (Daskalakis-Ilyas-Syrgkanis-Zeng 2018). Witness golden file shows the cycle of OGD's last-iterate vs. the convergence of OGDA's last-iterate over identical 10⁴-step trajectories — single most-cited result of the 2018-2020 min-max-optimisation literature.

### PR-4 — CFR family (~520 LOC, two days)
G10 ExtensiveForm interface + G11 CFR/CFR+/LinearCFR/DiscountedCFR + G9 EXP3 + G12 MCCFR.
- Composes G6 RegretMatching at every information set of G10's tree. Each CFR variant differs only in iteration weighting / regret-truncation — same core, different decorations. CFR+ uses RM+ from PR-2.
- Cross-validation: G11 with RM+ on a small Kuhn-poker tree converges to ε-Nash within 10⁵ iterations; G12 MCCFR with outcome-sampling matches G11 to 1e-3 within 10⁶ samples — second R-MUTUAL pin on poker.
- Out-of-scope this PR: a real EF-game library. G10 ships only the abstract interface; consumers (poker, bridge, turn-based games) implement against it.

### PR-5 — equilibrium computation & bilevel (~470 LOC, two days)
G3 ReplicatorDynamics + G19 CorrelatedEquilibriumLP + G20 StackelbergBilevelLP.
- G3 forces first cross-edge `gametheory/ → chaos/` (plus `gametheory/ → optim/proximal/` already in PR-3).
- G19 saturates **R-CCE-FROM-NO-INTERNAL-REGRET 2/2 pin**: long-run average of G6 RegretMatching ↔ CCE LP solution agree to 1e-3 over 10⁵ rounds on a 3×3×3 game.
- G20 Stackelberg is the single biggest cross-link to RubberDuck's auction-strategy consumer named in `gametheory/nash.go:9-15`.

**Total: 5 PRs, ~2,080 LOC source + ~500 LOC tests, ~6.5 engineer-days.** Three R-MUTUAL-CROSS-VALIDATION pins land, two architectural witnesses (R-OMD-UNIFIES-FIVE, R-LAST-ITERATE-VS-AVERAGE).

---

## Precision hazards

- **Fictitious play averaging window.** `Minimax` at `nash.go:172` hard-codes `maxIter=100000` without examining convergence. G7 promotes this to public + adds residual stop criterion `‖p_avg − p_avg_prev‖_∞ < 1e-6` after a min iteration count — convergence is `O(1/√T)` so 100k iters → ≤3e-3 absolute, NOT 1e-6 (074-T2.5 confusion of asymptotic-rate vs absolute-tolerance).
- **Hedge step size η.** Theoretical optimum `η = √(8 log K / T)` requires known horizon `T`. For anytime variant use doubling trick or `ηₜ = √(log K / t)`. Document the trade-off; default to anytime.
- **Regret matching ties.** When all positive-cumulative-regrets are zero (start of run, or after RM+ truncation), uniform play is the canonical fallback (Hart-Mas-Colell 2000 fn. 4) — pin to this for golden-file determinism.
- **Replicator dynamics simplex drift.** RK4 on `ẋ = x ⊙ (Ax − xᵀAx·1)` is mathematically simplex-preserving (sum-derivative is zero) but numerically not exactly so over 10⁶ steps. `ProxSimplex` projection after each step bounds drift at machine epsilon; without it, drift compounds to ~1e-6 per million steps — flag `R-MUTUAL` floor at `T=10⁶`, not bug.
- **Optimistic OGD step size.** Daskalakis-Ilyas-Syrgkanis-Zeng (2018) require `η < 1/(2L)` (vs. `η < 1/L` for OGD) for the `O(1/T)` last-iterate guarantee. Default `η = 1/(4L)` (5x conservative) and document.
- **Extragradient `F` Lipschitz.** Same `1/(2L)` step bound; if `L` is unknown, expose Lipschitz-estimation closure on caller (mirror `optim.LBFGS` line-search abstraction).
- **CFR regret + RM+ negative cap.** RM+ truncates `R⁻ᵢ ← 0` per iteration — the truncation IS the algorithm; without it CFR+ degrades to vanilla CFR. Tammelin-Burch-Johanson-Bowling (2014) section 4 explicit on this.
- **Lemke-Howson degenerate games.** Lexicographic perturbation pin: use `(b₁, ε, ε², …, εⁿ)` symbolic, never a fixed `ε > 0` — von Stengel ch. 3 §3.5 on numerical Lemke. 072 T1-3 already names this.
- **Frank-Wolfe sublinear floor.** FW achieves `O(1/T)` for smooth convex on compact `K` but only `O(1/√T)` without smoothness — document and recommend Nesterov-Frank-Wolfe (Lacoste-Julien-Jaggi 2015 away-step variant, deferred to v1.x).
- **EXP3 importance-weight variance blowup.** `r̂ᵢₜ = rᵢₜ/pᵢₜ·𝟙[aₜ=i]` has variance ~1/min(p). Min-prob mixing `pᵢ ← (1−γ)pᵢ + γ/K` standard fix (Auer et al. 2002 EXP3 with γ=√(K log K / T)).
- **CCE-LP scale.** For `n` players each with `mᵢ` actions, joint distribution has `Πmᵢ` variables → LP grows multiplicatively. `optim.SimplexMethod` Bland-rule worst-case is exponential; use `optim.InteriorPoint` for `n≥3` (acknowledge 102 audit's flag that current `InteriorPoint` is barrier-gradient not Newton-on-KKT — quality-of-answer hazard, not correctness).

---

## Architectural placement

**Consumer-side, in `gametheory/`** — 16 consecutive synergies (158/159/160/161/165/166/167/168/169/170/171/172/173 + this) confirm consumer-side placement is the codebase convention. New files:

```
gametheory/
  online.go              # G4 interface, G5 Hedge, G6 RegretMatching/+, G7 FP-public, G8 SmoothFP, G9 EXP3, G14 OMD
  zerosum.go             # G1 ZeroSumViaSimplex, G2 LemkeHowsonBimatrix
  replicator.go          # G3 ReplicatorDynamics
  oco.go                 # G13 OGD, G15 FTRL, G16 OptOGD, G17 Extragradient, G18 FrankWolfe
  extensive.go           # G10 EF interface
  cfr.go                 # G11 CFR/CFR+/LinearCFR/DiscountedCFR, G12 MCCFR
  equilibrium.go         # G19 CCE-LP, G20 Stackelberg-bilevel
```

Cycle-free DAG: `gametheory/ → {optim/, optim/proximal/, chaos/, prob/}`. Reverse direction never. Verified by enumeration; mirrors prior 13 synergies.

The private fictitious-play loop at `gametheory/nash.go:170-209` STAYS — `Minimax` is its public face — but the loop body is extracted to `gametheory/online.go`'s `FictitiousPlay.Step()` and the `Minimax` function-body becomes a 5-line driver over `FictitiousPlay`. PR-1 includes this refactor; the public `Minimax` API does not change (one of the strongest signals that this synergy is overdue).

---

## Cross-package dependency direction (confirmed cycle-free)

```
gametheory/ ──→ optim/                 (G1, G2, G15, G18, G19, G20)
gametheory/ ──→ optim/proximal/        (G3, G13, G16, G17)
gametheory/ ──→ optim/transport/       (none today; dual-Sinkhorn = OMD with KL on transport polytope is candidate v1.x)
gametheory/ ──→ chaos/                 (G3 only; chaos.RK4Step)
gametheory/ ──→ prob/                  (G5, G8 only; prob.SoftmaxStable, no new edge if 169-S8 LogPDF lands first)
optim/      ──→ gametheory/            (none; never reverses)
chaos/      ──→ gametheory/            (none)
```

All edges new in this PR sequence. Zero edges between any pair before PR-1. After PR-5, gametheory has 4-5 outbound edges, all to packages that themselves have stable dependency positions.

---

## Distinct-from-prior-reviews provenance

This is the **fourteenth Block-B synergy review** in the 174-of-400 sequence and the **first** `gametheory × optim` review.

- **071-075 (gametheory isolation).** 071 §F1 names "fictitious play embedded in `Minimax` should be public no-regret iterator". 072 T1-3 names Lemke-Howson at ~300 LOC. 072 T1-7 names "regret-matching + Hedge / MWU + smooth fictitious play" at ~150 LOC. 072 T2-1 names ReplicatorDynamics at ~120 LOC chaos-coupled. **THIS review composes those isolation gaps with `optim.SimplexMethod` and `optim/proximal.ProxSimplex` — neither named cross-package composition, only in-package gaps.**
- **101-105 (optim isolation).** 102 T1.6-T1.8 names Adam/RMSprop/SGD-Nesterov as the missing modern optimisers; 102 T2.21 names BO-EI. **THIS review touches none of those — it composes around the existing nine optim algorithms.**
- **073 gametheory-sota.** Names nashpy/Gambit/OpenSpiel as Tier-1 cross-language pinning targets. **THIS review's PRs land the algorithm set (CFR family, Lemke-Howson, replicator, fictitious play, no-regret) that those references all ship.**
- **142 topology-missing.** Names BoltzmannEntropy on simplex (T1.6) — orthogonal axis but consumes the same `proximal.ProxSimplex` projector that G3 ReplicatorDynamics and G13/G16 OGD-on-simplex consume; cross-link.
- **163 synergy-optim-autodiff.** First-cousin synergy. 163 ships forward-mode duals + HVP + Newton-CG + Wolfe + Adam infrastructure on the optim+autodiff axis. **THIS review is strictly orthogonal**: gametheory×optim composes first-order online updates over the existing optim primitives without needing autodiff at all (every G1-G20 gradient is hand-derivable closed-form against the bilinear payoff). If PR-2 of 163 lands first, G15 FTRL gains the option of reverse-mode-AD-supplied gradient closure to the inner-argmin `LBFGS` — but that is decoration, not necessity.
- **169 synergy-prob-optim.** S15 BBVI uses score-function gradient over distributions. **THIS review's G14 OMD with negative-entropy mirror map ≡ Hedge ≡ entropic exponential-family natural gradient on simplex** — pure cross-link to 169-S15 BBVI's reparameterised gradient on Categorical, no shared primitive.
- **170 synergy-info-prob.** S6 MINE/InfoNCE variational MI bounds — orthogonal axis but consumes G18 FrankWolfe candidate-vertex selection in finite-event-space MI estimation; cross-link only.
- **154 synergy-chaos-timeseries.** Orthogonal forecasting axis; G3 ReplicatorDynamics shares the chaos.RK4Step consumer-side-placement precedent.
- **161 synergy-control-prob.** Orthogonal Lyapunov axis; G15 FTRL with quadratic regulariser on quadratic loss reduces to Kalman-update-style Riccati recursion — cross-link only.

**Why this synergy is not yet reviewed:** all four prerequisites — gametheory isolation (071-075), optim isolation (101-105), proximal sub-package shipping (current `optim/proximal/`), and the prior 13 synergies establishing consumer-side-placement convention — needed to be in place. This is the first review where (a) `optim.SimplexMethod` is documented and stable (b) `optim/proximal.ProxSimplex` is shipped (c) the private FP loop's promotion-target abstraction (`OnlineLearner`) is justified by the breadth of consumers (G5-G18) (d) the cycle-free direction is settled. Bottom line: this is **the highest-leverage synergy not yet reviewed** because (i) it requires only one new abstraction (40-LOC interface), (ii) it lifts a private loop that is already cited in three different audits as needing public promotion, (iii) the topic-prompt's "LP for zero-sum (use optim.SimplexMethod)" is a literally-zero-LOC composition that has not been written, (iv) three R-MUTUAL-CROSS-VALIDATION pins fall out for free, (v) 5 PRs over ~6.5 engineer-days lands fourteen of fifteen named topic-prompt items against v0.10.0.

Report at `agents/174-synergy-gametheory-optim.md`, ~310 lines.

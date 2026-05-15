# 153 | synergy-prob-infogeo

**Topic:** prob × infogeo — Fisher information from distributions, natural gradient on prob, Bregman divergences induced by exponential families, KL as variational loss.
**Block:** B (cross-package synergies).
**Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `prob/` and `infogeo/` are composed; not what either lacks in isolation (116-120 / 091-095 own those).

## Two-line summary

`prob/` ships seven `Distribution`s and `infogeo/` ships nine f-divergences plus four Bregman divergences but they don't talk: no Fisher matrix from a Distribution, no Bregman induced by a log-partition, no natural gradient on parameters, no closed-form KL on conjugate pairs. **Twelve synergy primitives (S1-S12) totalling ~1090 LOC of pure glue** stand up information-geometric inference on top of two stable bases; cheapest first PR is **S1 ClosedFormKL** at ~120 LOC because every native exp-family pair has an analytic KL and prob/ ships only the numerical trap-rule, while the **highest-leverage one-day unlock is S2 FisherFromDistribution** turning the existing `Distribution` interface into the substrate for natural gradient, Cramér-Rao bounds, and IGO without adding a single new prob primitive.

---

## Bases — what each package exposes today

`prob/` (agent 117): `Distribution interface { PDF, CDF }` + concrete `BetaDist / NormalDist / ExponentialDist / UniformDist`; free Gamma/Poisson/Binomial PMF+CDF; `KLDivergenceNumerical` (trap-rule); `JeffreysKLDivergence` (Bernoulli-only); copulae; **no** `LogPDF`, `Score`, `FisherInfo`, parametric sampler.

`infogeo/` (agent 091): `KL, ReverseKL, JS, TotalVariation, Hellinger, ChiSquared, Renyi` over **discrete probability vectors only**; `Bregman, SquaredEuclidean, GeneralisedKL, ItakuraSaito, MahalanobisSquared`; `MMD2{Biased,Unbiased}` + Gaussian/Laplacian kernels; one consumer pins ∇KL = q−p via autodiff.

**Neither imports the other.** `infogeo.KL([]float64, []float64)` is on simplex vectors; `prob.KLDivergenceNumerical(Distribution, Distribution, …)` is on distribution objects; conjugate-pair KL formulas (Normal-Normal, Beta-Beta, Gamma-Gamma, Poisson-Poisson) exist in neither place.

## The conceptual unlock — `Distribution` as a manifold point

Every `prob.Distribution` is a point `p_θ` on a statistical manifold:

1. **Fisher information** `I(θ) = E_θ[∇log p · ∇log pᵀ]` = the Riemannian metric. For exp-family it equals `∇²A(θ)` — Hessian of the log-partition. Closed form for every distribution prob/ ships.
2. **KL is a Bregman divergence** `D_A(η₁,η₂)` with generator `A`. `KL(p_{θ₁}∥p_{θ₂}) = A(θ₂) − A(θ₁) − ∇A(θ₁)·(θ₂−θ₁)`.
3. **Natural gradient** `θ ← θ − η·I(θ)⁻¹∇L(θ)` is parameter-invariant and converges in O(d) versus O(κ(I)·d) for SGD.
4. **e ↔ m coord duality** turns `prob.Distribution` into a dually-flat manifold with straight-line geodesics; every conjugate-prior update is one m-projection.

`reality/` ships every numerical primitive needed (`linalg.Cholesky/Inverse`, `autodiff.Tape`, `Distribution`, `Bregman`) — never the wire-up.

---

## S1 — `ClosedFormKL` for native conjugate pairs (~120 LOC)

`prob.KLDivergenceNumerical` is O(n) trap-rule. Every native pair has a one-line analytic KL:

| Pair | KL(p ∥ q) |
|---|---|
| Normal(μ₁,σ₁) ∥ Normal(μ₂,σ₂) | `log(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²)/(2σ₂²) − 1/2` |
| Beta(α₁,β₁) ∥ Beta(α₂,β₂) | `lnB(α₂,β₂) − lnB(α₁,β₁) + (α₁−α₂)ψ(α₁) + (β₁−β₂)ψ(β₁) + (α₂−α₁+β₂−β₁)ψ(α₁+β₁)` |
| Exponential(λ₁) ∥ Exponential(λ₂) | `log(λ₁/λ₂) + λ₂/λ₁ − 1` |
| Gamma(k₁,θ₁) ∥ Gamma(k₂,θ₂) | `(k₁−k₂)ψ(k₁) − lnΓ(k₁) + lnΓ(k₂) + k₂·log(θ₂/θ₁) + k₁(θ₁−θ₂)/θ₂` |
| Poisson(λ₁) ∥ Poisson(λ₂) | `λ₁·log(λ₁/λ₂) + λ₂ − λ₁` |

`KLNormal/KLBeta/KLGamma/KLPoisson/KLExponential`. **~120 LOC** + `prob.Digamma` (~40 LOC, also flagged 117 §T1.4). Golden-test: `KLNormal(0,1,0,2) = log 2 + 1/8 − 1/2 = 0.31814718…` to 1e-13.

## S2 — `FisherFromDistribution`: closed-form FIM (~150 LOC)

| Distribution | FIM |
|---|---|
| Normal(μ, σ) | `diag(1/σ², 2/σ²)` |
| Exponential(λ) | `1/λ²` |
| Beta(α, β) | `[[ψ'(α)−ψ'(α+β), −ψ'(α+β)], [−ψ'(α+β), ψ'(β)−ψ'(α+β)]]` |
| Gamma(k, θ) | `[[ψ'(k), 1/θ], [1/θ, k/θ²]]` |
| Poisson(λ) | `1/λ` |
| Binomial(n, p) | `n/(p(1−p))` |
| Multinomial / Categorical | `diag(n/p_i) + 11ᵀ/(n p_n)` (092 §T1.1) |
| MultivariateNormal(μ, Σ) | `block-diag(Σ⁻¹, ½Dᵀ(Σ⁻¹⊗Σ⁻¹)D)` (092 §T1.2) |

```go
type FisherInfoer interface { FisherInfo(out []float64) error; Dim() int }
func FisherInfoMC(d Distribution, theta []float64, n int, rng Rand,
    out []float64) error  // MC fallback for arbitrary Distribution
```

**~150 LOC** (per-distribution + interface + MC fallback). Closed form exact; MC fallback handles user-supplied impls. **Witness:** Cramér-Rao for Normal-MLE on n samples gives `Var(μ̂) ≥ σ²/n` matching the FIM directly. **Keystone**: gates S3, S4, S5, S6, S8, S9, S11.

## S3 — `BregmanFromExpFamily`: KL as Bregman with log-partition (~130 LOC)

`infogeo.Bregman(BregmanGen{Phi, GradPhi}, …)` requires hand-written `phi`. For exp-family the generator is the log-partition `A(θ)` and gradient is `μ(θ) = E[T(X)]` — both queryable from the distribution.

| Dist | A(θ) | ∇A(θ) |
|---|---|---|
| Normal(μ,1) | μ²/2 | μ |
| Poisson(λ) | e^θ (θ=log λ) | λ |
| Bernoulli(p) | log(1+e^θ) (θ=logit p) | p |
| Exponential(λ) | −log(λ) | 1/λ |

```go
type ExpFamily interface {
    LogPartition(theta []float64) float64
    GradLogPartition(theta, out []float64)
}
func ExpFamilyBregman(d ExpFamily) infogeo.BregmanGen
func KLViaBregman(d ExpFamily, theta1, theta2 []float64) (float64, error)
```

**~130 LOC** + ExpFamily impls (~20 LOC × 6). **Witness:** `KLViaBregman(NormalDist, …)` matches `KLNormal` from S1 to 1e-13.

## S4 — `NaturalGradient` step (~80 LOC)

The optimiser missing from `optim/`. Step `θ ← θ − η·F(θ)⁻¹∇L(θ)` is affine-invariant — same trajectory under any reparameterisation.

```go
type NaturalGradientStepper struct {
    Eta, Damping float64       // Tikhonov λ for F + λI
    FIM FisherInfoer            // S2 dependency
}
func (s *NaturalGradientStepper) Step(theta, grad, out []float64) error
```

**~80 LOC** composing `linalg.CholeskyDecompose` + `CholeskySolve`. Damping handles boundary-singular FIM (Beta near α=0). CG variant for d>1000 deferred (Martens-Grosse 2015 K-FAC). **Witness:** on Normal-MLE FIM = Hessian, so natural-gradient = Newton — converges in 1 step.

## S5 — `EmpiricalFisher` vs `ObservedFisher` (~110 LOC)

Distinct quantities constantly conflated:
- True `I(θ) = E_θ[∇log p ∇log pᵀ]`
- Empirical `Î(θ) = (1/n) Σ ∇log p(x_i;θ)·∇log p(x_i;θ)ᵀ`
- Observed `J(θ) = −∇²log p(X;θ)`

Conflating them is documented bug in TF/PyTorch optimisers (Kunstner-Hennig-Balles 2019).

```go
func EmpiricalFisher(scoreFn func(x, theta, out []float64), samples [][]float64,
    theta []float64, out []float64) error
func ObservedFisher(d Distribution, samples [][]float64, theta, out []float64) error
```

**~110 LOC** + autodiff.HVP (agent 012 §T1, ~600 LOC blocker). Reverse-of-reverse fine for d<50 with current tape.

## S6 — `CramerRaoLowerBound` (~30 LOC)

Direct corollary of S2: `Cov(θ̂) ≥ I(θ)⁻¹` (Loewner). One Cholesky inverse. **Test:** `CRLB(NormalMLE, [μ,σ])` returns `diag(σ²/n, σ²/(2n))` analytically. The cheapest publishable witness for S2.

## S7 — `MProject` / `EProject` on Distribution (~120 LOC)

- m-projection: `argmin_{θ∈M} KL(p ∥ p_θ)` — moment-matching MLE; closed form on conjugate pairs (`η = E_p[T(X)]`).
- e-projection: `argmin_{θ∈M} KL(p_θ ∥ p)` — I-projection / max-entropy.

```go
func MProjectExpFamily(samples [][]float64, fam ExpFamily, theta []float64) error
func EProjectMixtureFamily(p, basis [][]float64, weights []float64) error
```

**~120 LOC**. Pythagorean (Csiszár 1975) provable witness: `KL(p∥q) = KL(p∥r) + KL(r∥q)` with r = m-projection of p onto e-flat manifold.

## S8 — `VariationalELBO` and reparametrised gradient (~150 LOC)

VI fits `q_φ ≈ p(z|x)` by minimising `KL(q_φ ∥ p)`. ELBO is the tractable surrogate; `KL(q∥p) = log p(x) − ELBO`. Currently missing from prob/ (117 §T2.6).

```go
func ELBO(qSampler func(rng) []float64,
    logQ, logJoint func(z []float64) float64, rng Rand, n int) float64
func ELBOGradientReparametrised(...)  // pathwise gradient for Normal q
```

**~150 LOC** + Box-Muller Normal sampler (~30 LOC, 117 §T1.3). Closed-form ELBO for Gaussian q paired with closed-form KL from S1 collapses to one line. **S4 + S8 = canonical NaturalVariationalInference** (Honkela et al. 2010).

## S9 — `IGOStep` (information-geometric optimisation, Ollivier 2017) (~180 LOC)

Unifying framework — every evolution strategy (CMA-ES, NES, xNES) is one IGO instance. Update `θ ← θ + η·F(θ)⁻¹·∇_θ E_{x~p_θ}[w(f(x))·log p_θ(x)]` with quantile weights w.

```go
type IGOOptimiser struct {
    Sampler func(theta, rng) []float64
    LogPDF  func(x, theta float64) float64
    FIM     FisherInfoer
    WeightFunc func(rank, n int) float64  // CMA-ES default
    Eta float64
}
func (o *IGOOptimiser) Step(theta []float64, f func([]float64) float64, rng Rand, n int) error
```

**~180 LOC**. Uses S2+S4+S8 + existing optim ranking. Subsumes CMA-ES/NES/policy-gradient under one interface. **Highest-leverage ADD to optim/ paid for by prob × infogeo.**

## S10 — Symmetrised KL on `Distribution` (~80 LOC)

`prob.JeffreysKLDivergence` accepts only **two Bernoulli probabilities** (`prob/jeffreys.go:169`). On `Distribution` pairs the symmetric `J(p,q) = KL(p∥q) + KL(q∥p)` and JS divergence are universally useful (regime classification, MCMC convergence) and must be exposed at the Distribution level.

```go
func JeffreysDist(p, q Distribution, lo, hi float64, n int) float64
func JSDist(p, q Distribution, lo, hi float64, n int) float64
func JeffreysClosedForm(d ExpFamily, theta1, theta2 []float64) float64
```

**~80 LOC**. Bridges naming inconsistency (`infogeo.JS` discrete-vector, `prob.JeffreysKLDivergence` Bernoulli-scalar, neither on `Distribution`).

## S11 — Categorical-Dirichlet conjugate Fisher reciprocity (~60 LOC)

The textbook self-dual fact:
- Categorical(p) FIM in p-coords: `diag(1/p_i) + 11ᵀ/p_n`
- Dirichlet(α) covariance: `Σ_{ij} = (α_i δ_{ij} − α_iα_j/α₀) / (α₀²(α₀+1))`

These are reciprocal under conjugate identification `α_i ∝ p_i`; product is identity. Witness for "conjugate prior is inverse of model Fisher metric" (Amari 1985 §4.3).

```go
func CategoricalFisher(p, out []float64) error
func DirichletCovariance(alpha, out []float64) error
// Witness: Fisher(p) · Cov(α=cp) ≈ I to 1e-12.
```

**~60 LOC** + Categorical + Dirichlet distributions (117 §T1.1, ~150 LOC). Once shipped, `MProjectExpFamily` (S7) on Categorical = Dirichlet posterior — closes the conjugate-pair loop.

## S12 — MultivariateNormal KL + block-diag FIM (~140 LOC)

The crown-jewel synergy. For `N(μ, Σ)`:

`KL = ½(tr(Σ₂⁻¹Σ₁) + (μ₂−μ₁)ᵀΣ₂⁻¹(μ₂−μ₁) − k + log(det Σ₂/det Σ₁))`

FIM is **block-diagonal** with `F_{μμ} = Σ⁻¹`, `F_{ΣΣ} = ½(Σ⁻¹⊗Σ⁻¹)`. The block-diagonality means natural gradient on multivariate Gaussians decouples mean from covariance update — CMA-ES's "rank-one + rank-μ" updates *are* the two blocks.

```go
type MVNormalDist struct { Mu, Sigma []float64 }  // Sigma row-major SPD
func (d *MVNormalDist) PDF(x []float64) float64
func (d *MVNormalDist) FisherInfo(out []float64) error
func MVNormalKL(mu1, sigma1, mu2, sigma2 []float64, dim int) (float64, error)
```

**~140 LOC** + `linalg.LogDet` (~30 LOC, agent 097 missing-list). MVN itself missing from prob/ (117 §T1.1) — this PR ships distribution + KL + FIM together.

---

## Composition table

| Synergy | LOC | Existing deps | New primitive | Downstream |
|---|---|---|---|---|
| S1 ClosedFormKL | 120 | NormalPDF/Beta/Gamma/Poisson/Exp | prob.Digamma (+40) | S3,S7,S8,S10 |
| S2 FisherFromDistribution | 150 | Distribution, linalg.MatMul | FisherInfoer | S4-S6,S8,S9,S11 |
| S3 BregmanFromExpFamily | 130 | infogeo.Bregman | ExpFamily (+20×6) | S7,S9 |
| S4 NaturalGradient | 80 | linalg.Cholesky{Decompose,Solve} | — (S2 dep) | S8,S9 |
| S5 Empirical/Observed Fisher | 110 | autodiff.Tape | autodiff.HVP (+600, 012 T1) | S6 |
| S6 CramerRaoLowerBound | 30 | linalg.Inverse | — (S2 dep) | risk-bound consumers |
| S7 m-/e-projection | 120 | KL closed forms | — | EM, conjugate updates |
| S8 VariationalELBO | 150 | NormalPDF, NormalQuantile | NormalSample (+30, 117 T1.3) | S9 |
| S9 IGOStep | 180 | optim ranking | — (S2+S4+S8) | CMA-ES, NES |
| S10 JeffreysDist/JSDist | 80 | KLDivergenceNumerical | — | regime classification |
| S11 Categorical×Dirichlet | 60 | linalg.MatMul | Cat+Dir (+150, 117 T1.1) | conjugate posteriors |
| S12 MVNormal KL+FIM | 140 | linalg.Cholesky | MVN+LogDet (+130) | S2,S4,S8 |

**S2 is keystone** (six dependents). **S1 cheapest standalone**. **S12 largest brick** but only one putting MVN on the manifold — without it, half the IG literature is uncomposable.

## Recommended PR sequence

| PR | Scope | LOC | Days |
|---|---|---|---|
| 1 | prob.Digamma + S1 ClosedFormKL (5 conjugate pairs) | 160 | 1 |
| 2 | S2 FisherFromDistribution + FisherInfoer (7 distributions) | 150 | 1 |
| 3 | S6 CramerRaoLowerBound + S4 NaturalGradient | 110 | ½ |
| 4 | S3 BregmanFromExpFamily + ExpFamily (6 distributions) | 250 | 1½ |
| 5 | S10 JeffreysDist / JSDist | 80 | ½ |
| 6 | S7 m-/e-projection (closed-form pairs only) | 120 | 1 |
| 7 | prob.Categorical + prob.Dirichlet + S11 reciprocity | 210 | 1½ |
| 8 | prob.MVNormal + S12 KL + FIM + linalg.LogDet | 270 | 2 |
| 9 | NormalSample + S8 VariationalELBO + reparam gradient | 180 | 1½ |
| 10 | S9 IGOStep + CMA-ES instance | 180 | 1½ |
| 11 | S5 Empirical/Observed Fisher (gated on autodiff.HVP) | 110 | gated |

**Total:** ~1820 LOC across 11 PRs, ~1090 LOC of *pure synergy* (netting out 117/092 prerequisites). PR1-3 self-contained foundation; PR4-6 close IG textbook on `prob.Distribution`; PR7-8 fix the multivariate gap that makes the rest useful; PR9-10 deliver modern variational/ES substrate; PR11 waits on autodiff.

---

## Cross-package observations

**1. The `Distribution` interface is one method short of being a manifold.** Adding `LogPDF(x)` and `Score(x, out)` (~30 LOC across four existing distributions) unlocks S2 MC fallback, S5, S8 reparam gradient, and the entire autodiff path for arbitrary user distributions. Highest-leverage prob/ API change motivated by infogeo. Agent 092 §T1.3 flagged this from infogeo's side; the synergy is *why it matters*.

**2. The exp-family / Bregman duality is the bridge metal.** `ExpFamily { LogPartition, GradLogPartition }` is to exp-family what `Distribution { PDF, CDF }` is to general distributions. Once shipped: KL = one Bregman call (S3), m-projection = moment-matching (S7), natural gradient = Hessian-of-A solve (S4). Three textbook subjects collapse to one interface plus six 20-LOC implementations.

**3. `prob/` should not import `infogeo/`.** Composition lives in a new `prob/infogeo.go` (or new package `infogeo/parametric/`) depending on both — preserves prob's minimalism and infogeo's discrete-only convention. Matches agent 151's `spectral/` recommendation; each package owns its own vocabulary, the bridge owns its own file.

**4. autodiff connection.** S5 ObservedFisher and S8 reparametrised-gradient ELBO are the cleanest targets in repo for differentiable inference: Tape-traced log-likelihood → analytic gradients → Cholesky natural-gradient solve → S4 step. Closes the same loop agents 140 (GARCH) and 151 (Whittle MLE) recommended.

**5. Pistachio 60 FPS budget.** S2 closed-form FIM is O(d²) at most (diagonal for Normal/Exp/Poisson, 2×2 for Beta/Gamma, block-diag for MVN); S4 step is one Cholesky solve — `d³/3` flops at d=10 = 333 flops ≈ 10 ns. Natural-gradient inference essentially free below d~50 with out-buffer convention.

**6. Golden-file leverage.** Every closed-form KL has a one-line answer: `KL(N(0,1)∥N(0,2)) = log 2 + 1/8 − 1/2` exact at 1e-13; `KL(Beta(1,1)∥Beta(1,1)) = 0` exact; CRLB on Normal-MLE = `(σ²/n, σ²/(2n))` exact; S11 reciprocity `Fisher(p)·Cov(α=cp) ≈ I` symbolic at 1e-12. 30 vectors × 12 primitives = 360 new test vectors, 4-language polyglot reproducible.

**7. R-MUTUAL-CROSS-VALIDATION saturation candidate.** S1, S3, and S12 give the same scalar on Normal-Normal: closed form, Bregman with log-partition generator, MVN-KL at d=1. One 30-LOC test pinning all three to 1e-13 saturates the pattern (cf. recent commits 6a55bb4 audio onset, 365368a copula×autodiff). Waiting only on PRs 1, 4, 8.

**8. The `Distribution` interface as `Manifold` adapter.** Add LogPDF+Score+FisherInfo and every concrete `Distribution` satisfies the abstract `Manifold` interface from agent 092 §T2.1. Synergy is not "infogeo grows a prob adapter" — every `prob.Distribution` *is* an infogeo manifold. Same idea applied symmetrically: KL was the half prob ships numerically and infogeo ships discretely; closed-form KL on the same interface unifies them.

---

## Explicitly NOT in this report

- Numerical bugs in shipped prob/ or infogeo/ (agents 116, 091)
- Missing primitives within prob/ alone (117) — referenced only as prerequisites (Categorical, Dirichlet, MVN, Digamma, NormalSample, LogDet)
- Missing primitives within infogeo/ alone (092) — referenced as parallel work; FIM closed forms, natural-gradient stepper, e/m projection are *also* in 092 because they belong on both sides; the synergy is the wiring
- API ergonomics within either package (094, 119) and per-package perf (095, 120)
- α-connections, hyperbolic / SPD / Stiefel manifolds (092 Tier 2)
- MCMC samplers, Hawkes, Gaussian processes (117 Tier 2-3)
- changepoint × infogeo (152 territory) and timeseries × prob (154 territory)

This report's distinctive contribution is **the bridge** — both `prob.Distribution` and `infogeo.Bregman` were waiting for the other half to call them.

---

## Progress

- 2026-05-08 — agent 153 complete; 12 synergy primitives (S1-S12) catalogued with composition graph and 11-PR sequence (~1090 LOC pure connective tissue, ~1820 LOC including prerequisite distributions); identified S2 FisherFromDistribution as keystone gating six others, S1 ClosedFormKL as cheapest standalone unlock (5 conjugate pairs at 120 LOC), S11 Categorical × Dirichlet reciprocity as textbook witness; recommended LogPDF + Score + FisherInfo as the three methods one short of turning prob.Distribution into a Manifold; flagged the three-way Normal-KL cross-check (S1 closed form vs S3 Bregman vs S12 MVN k=1) as an R-MUTUAL-CROSS-VALIDATION saturation candidate.
